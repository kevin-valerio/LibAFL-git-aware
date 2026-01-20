#!/usr/bin/env bash
set -euo pipefail

DEFAULT_TRIALS=10
DEFAULT_BUDGET_SECS=60
DEFAULT_WARMUP_SECS=30
DEFAULT_RUST_TOOLCHAIN="1.90.0"

# Benchmark overview (high level)
#
# Goal: compare baseline scheduling vs git-aware scheduling on a target where the crash is in
# "recently changed" code (as determined by `git blame`).
#
# Flow:
#  1) Create a temporary `reth` git checkout under $bench_root/reth (shallow clone).
#  2) Add a small in-process fuzzer crate: crates/libafl-gitaware-bench (writes queue/ + crashes/).
#  3) Create a synthetic "old" baseline commit (dated 2000-01-01) where RECENT_BUG does NOT crash.
#  4) Build the baseline binary with:
#       - SanitizerCoverage trace-pc-guard edges
#       - LibAFL git-recency LLVM plugin (records debug locations for pcguard indices)
#  5) Warmup: run the baseline binary for --warmup seconds (mapping disabled) to generate an initial
#     corpus in $bench_root/warmup/out/queue. This reduces variance by avoiding "startup exploration"
#     dominating the results, and ensures both variants start from identical inputs.
#  6) Introduce a new commit that flips a single line to crash (marked "RECENT_BUG") and rebuild.
#     Because it's a new commit, `git blame` will mark that line as "recent".
#  7) Generate the `pcguard_index -> git blame timestamp` mapping file for the rebuilt binary.
#  8) Run paired trials (baseline vs git-aware), each starting from the warmup corpus:
#       - baseline: do NOT set LIBAFL_GIT_RECENCY_MAPPING_PATH (no recency boost)
#       - git-aware: set LIBAFL_GIT_RECENCY_MAPPING_PATH (recency boost enabled)
#     Time-to-first-crash is measured by watching for a file to appear in out/crashes/.
#
# Output layout:
#   $bench_root/in/                  initial seeds
#   $bench_root/tools/               built libafl tools + git-recency LLVM plugin
#   $bench_root/warmup/out/queue/    warmup-generated initial corpus
#   $bench_root/runs/{baseline,git-aware}/trial_*/out/{queue,crashes}
#
usage() {
  cat <<'EOF'
Usage:
  scripts/git_aware_reth_bench.sh [--trials N] [--budget SECS] [--warmup SECS] [--bench-root DIR]
EOF
  cat <<EOF

Defaults:
  --trials     ${DEFAULT_TRIALS}
  --budget     ${DEFAULT_BUDGET_SECS}
  --warmup     ${DEFAULT_WARMUP_SECS}
  --bench-root \$BENCH_ROOT or a mktemp dir under /tmp
  RUST_TOOLCHAIN=\$RUST_TOOLCHAIN or ${DEFAULT_RUST_TOOLCHAIN}
EOF
  cat <<'EOF'

What it does:
  - Creates a local benchmark repo in /tmp based on a shallow checkout of:
      https://github.com/paradigmxyz/reth
  - Adds a LibAFL in-process fuzzer crate + a "recent" crashing line (RECENT_BUG)
    committed on top (so `git blame` marks it as newly-added/edited code).
  - Builds the fuzzer with Rust SanitizerCoverage (trace-pc-guard) and the LibAFL
    git-recency LLVM pass.
  - Generates a `pcguard_index -> git blame timestamp` mapping file.
  - Runs N paired trials (baseline vs git-aware) and prints median time-to-first-crash.

How warmup works:
  - Warmup runs the *baseline snapshot* (no crash) for --warmup seconds.
  - Warmup always disables the mapping (baseline behavior) and uses LIBAFL_RAND_SEED=0.
  - The resulting queue corpus directory:
      $bench_root/warmup/out/queue
    is then used as the input corpus for all trials.

How trials work:
  - After warmup, the script commits a single-line crash marked "RECENT_BUG" and rebuilds.
  - For each trial i=1..N, it runs the same fuzzer binary twice:
      1) baseline  (LIBAFL_RAND_SEED=i, mapping disabled)
      2) git-aware (LIBAFL_RAND_SEED=i, mapping enabled)
  - time-to-first-crash is wall-clock time until a file appears under out/crashes/.
    If no crash is found within --budget seconds, that trial counts as "budget" seconds.

Notes:
  - This script creates an intentionally crashing repo. Never push it anywhere.
  - It may install build deps (clang/llvm 20) and a rust toolchain via rustup.
EOF
}

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

mktemp_dir() {
  mktemp -d "/tmp/libafl_gitaware_reth_bench.XXXXXXXX"
}

rust_toolchain() {
  echo "${RUST_TOOLCHAIN:-${DEFAULT_RUST_TOOLCHAIN}}"
}

ensure_host_deps() {
  if ! command -v clang-20 >/dev/null 2>&1 || ! command -v llvm-config-20 >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y llvm-20-dev clang-20 lld-20
  fi

  if ! command -v rustup >/dev/null 2>&1; then
    echo "Missing rustup; please install rustup first." >&2
    exit 1
  fi

  local tc
  tc="$(rust_toolchain)"
  if ! rustc "+${tc}" -vV >/dev/null 2>&1; then
    rustup toolchain install "${tc}" -c rustc -c cargo -c rust-std
  fi
}

build_libafl_tools() {
  local libafl_root="$1"
  local out_dir="$2"
  local tc
  tc="$(rust_toolchain)"

  (
    cd "${libafl_root}"
    LLVM_CONFIG=llvm-config-20 cargo "+${tc}" build -p libafl_cc --release
  )

  local mapgen_bin="${libafl_root}/target/release/libafl_git_recency_mapgen"
  if [[ ! -x "${mapgen_bin}" ]]; then
    echo "Missing mapgen binary at ${mapgen_bin}" >&2
    exit 1
  fi

  local pass_so
  pass_so="$(find "${libafl_root}/target/release" -name 'git-recency-pass.so' -type f | head -n 1)"
  if [[ -z "${pass_so}" ]]; then
    echo "Failed to locate git-recency-pass.so under ${libafl_root}/target/release" >&2
    exit 1
  fi

  mkdir -p "${out_dir}"
  cp -f "${pass_so}" "${out_dir}/git-recency-pass.so"
  cp -f "${mapgen_bin}" "${out_dir}/libafl_git_recency_mapgen"
}

write_bench_crate() {
  local reth_root="$1"
  local libafl_root="$2"

  local crate_dir="${reth_root}/crates/libafl-gitaware-bench"
  mkdir -p "${crate_dir}/src"

  cat > "${crate_dir}/Cargo.toml" <<EOF
[package]
name = "libafl-gitaware-bench"
version = "0.0.0"
edition.workspace = true
publish = false

[dependencies]
libafl = { path = "${libafl_root}/crates/libafl", features = ["std"] }
libafl_bolts = { path = "${libafl_root}/crates/libafl_bolts", features = ["std"] }
libafl_targets = { path = "${libafl_root}/crates/libafl_targets", default-features = false, features = ["std", "coverage", "sancov_pcguard_edges"] }

alloy-primitives.workspace = true
reth-primitives-traits.workspace = true
reth-trie = { workspace = true, features = ["test-utils"] }
EOF

  cat > "${crate_dir}/src/main.rs" <<'EOF'
use core::time::Duration;
use std::{
    borrow::Cow,
    env,
    fs,
    path::{Path, PathBuf},
    process,
};

use libafl::{
    Error, HasMetadata,
    corpus::{CorpusId, InMemoryOnDiskCorpus, OnDiskCorpus},
    events::SimpleRestartingEventManager,
    executors::{ExitKind, inprocess::InProcessExecutor},
    feedbacks::{CrashFeedback, MaxMapFeedback},
    fuzzer::{Fuzzer, StdFuzzer},
    inputs::{BytesInput, HasTargetBytes},
    monitors::SimpleMonitor,
    mutators::{MutationResult, Mutator, StdMOptMutator, havoc_mutations},
    observers::{CanTrack, HitcountsMapObserver, TimeObserver},
    schedulers::{
        GitAwareStdWeightedScheduler, GitRecencyMapMetadata, IndexesLenTimeMinimizerScheduler,
        powersched::PowerSchedule,
    },
    stages::{calibrate::CalibrationStage, power::StdPowerMutationalStage},
    state::StdState,
};
use libafl_bolts::{
    AsSlice,
    current_time,
    Named,
    rands::StdRand,
    shmem::{ShMemProvider, StdShMemProvider},
    tuples::tuple_list,
};
use libafl_targets::std_edges_map_observer;

mod recent_bug;

#[derive(Clone, Copy)]
struct Cursor<'a> {
    data: &'a [u8],
    idx: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, idx: 0 }
    }

    fn take_u8(&mut self) -> u8 {
        if self.data.is_empty() {
            return 0;
        }
        let b = self.data[self.idx % self.data.len()];
        self.idx = self.idx.wrapping_add(1);
        b
    }

    fn take_u64(&mut self) -> u64 {
        let mut v = 0u64;
        for i in 0..8 {
            v |= (self.take_u8() as u64) << (i * 8);
        }
        v
    }

    fn take_20(&mut self) -> [u8; 20] {
        let mut out = [0u8; 20];
        for b in &mut out {
            *b = self.take_u8();
        }
        out
    }

    fn take_32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for b in &mut out {
            *b = self.take_u8();
        }
        out
    }
}

struct PreserveRethPrefixMutator<M> {
    name: Cow<'static, str>,
    inner: M,
}

impl<M> PreserveRethPrefixMutator<M> {
    fn new(inner: M) -> Self {
        Self {
            name: Cow::Borrowed("preserve_reth_prefix"),
            inner,
        }
    }
}

impl<M> Named for PreserveRethPrefixMutator<M> {
    fn name(&self) -> &Cow<'static, str> {
        &self.name
    }
}

impl<M, S> Mutator<BytesInput, S> for PreserveRethPrefixMutator<M>
where
    M: Mutator<BytesInput, S>,
{
    fn mutate(&mut self, state: &mut S, input: &mut BytesInput) -> Result<MutationResult, Error> {
        let had_prefix = input.as_ref().len() >= 4 && &input.as_ref()[..4] == b"RETH";
        let res = self.inner.mutate(state, input)?;
        if had_prefix && input.as_ref().len() >= 4 {
            input.as_mut()[..4].copy_from_slice(b"RETH");
        }
        Ok(res)
    }

    fn post_exec(&mut self, state: &mut S, new_corpus_id: Option<CorpusId>) -> Result<(), Error> {
        self.inner.post_exec(state, new_corpus_id)
    }
}

#[inline(never)]
fn bench_target(data: &[u8]) {
    use alloy_primitives::{keccak256, Address, B256, U256};
    use reth_primitives_traits::Account;

    let mut c = Cursor::new(data);

    // Build a small state snapshot and compute multiple trie roots.
    let n_accounts = 4usize + (c.take_u8() as usize % 8);

    let mut accounts = Vec::with_capacity(n_accounts);
    let mut accounts_prehashed = Vec::with_capacity(n_accounts);

    for _ in 0..n_accounts {
        let addr = Address::from_slice(&c.take_20());

        let nonce = c.take_u64();
        let balance = U256::from(c.take_u64());
        let account = Account { nonce, balance, bytecode_hash: None };

        let n_slots = 1usize + (c.take_u8() as usize % 4);
        let mut storage = Vec::with_capacity(n_slots);
        let mut storage_prehashed = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            let k = B256::from_slice(&c.take_32());
            let v = U256::from(c.take_u64());
            storage.push((k, v));
            storage_prehashed.push((keccak256(k.as_slice()), v));
        }

        accounts.push((addr, (account, storage)));
        accounts_prehashed.push((keccak256(addr.as_slice()), (account, storage_prehashed)));
    }

    let root_a = reth_trie::test_utils::state_root(accounts);
    let root_b = reth_trie::test_utils::state_root_prehashed(accounts_prehashed);

    let mut state = 0u64;
    state ^= u64::from_le_bytes(root_a.as_slice()[0..8].try_into().unwrap());
    state = state.rotate_left(17) ^ u64::from_le_bytes(root_b.as_slice()[0..8].try_into().unwrap());

    // Execute the "recent code" path (covered by inputs that start with b"RETH").
    recent_bug::maybe_trigger(data, state);
}

fn parse_args() -> (PathBuf, PathBuf, Duration) {
    let mut out: Option<PathBuf> = None;
    let mut input: Option<PathBuf> = None;
    let mut timeout_ms: u64 = 1200;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" | "--output" => out = Some(PathBuf::from(it.next().unwrap_or_default())),
            "-i" | "--input" => input = Some(PathBuf::from(it.next().unwrap_or_default())),
            "-t" | "--timeout" => {
                timeout_ms = it.next().unwrap_or_default().parse().unwrap_or(timeout_ms);
            }
            _ => {}
        }
    }

    let out = out.unwrap_or_else(|| {
        eprintln!("Missing -o/--output");
        process::exit(2);
    });
    let input = input.unwrap_or_else(|| {
        eprintln!("Missing -i/--input");
        process::exit(2);
    });

    (out, input, Duration::from_millis(timeout_ms))
}

fn fuzz(out_dir: PathBuf, seed_dir: &PathBuf, timeout: Duration) -> Result<(), Error> {
    let mut out_queue = out_dir.clone();
    let mut out_crashes = out_dir.clone();
    out_queue.push("queue");
    out_crashes.push("crashes");
    fs::create_dir_all(&out_queue)?;
    fs::create_dir_all(&out_crashes)?;

    let monitor = SimpleMonitor::new(|s| println!("{:?} {s}", current_time()));
    let mut shmem_provider = StdShMemProvider::new()?;
    let (state, mut mgr) =
        match SimpleRestartingEventManager::launch(monitor, &mut shmem_provider, true) {
            Ok(res) => res,
            Err(Error::ShuttingDown) => return Ok(()),
            Err(err) => return Err(err),
        };

    let edges_observer = unsafe { std_edges_map_observer("edges") };
    let edges_observer = HitcountsMapObserver::new(edges_observer).track_indices();
    let time_observer = TimeObserver::new("time");

    let mut feedback = MaxMapFeedback::new(&edges_observer);
    let mut objective = CrashFeedback::new();

    let rand_seed = env::var("LIBAFL_RAND_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let mut state = state.unwrap_or_else(|| {
        StdState::new(
            StdRand::with_seed(rand_seed),
            InMemoryOnDiskCorpus::new(out_queue).unwrap(),
            OnDiskCorpus::new(out_crashes).unwrap(),
            &mut feedback,
            &mut objective,
        )
        .unwrap()
    });

    if let Ok(mapping_path) = env::var("LIBAFL_GIT_RECENCY_MAPPING_PATH") {
        state.add_metadata(GitRecencyMapMetadata::load_from_file(mapping_path)?);
    }

    let calibration = CalibrationStage::new(&mut feedback);
    let mutator = StdMOptMutator::new(&mut state, havoc_mutations(), 7, 5)?;
    let mutator = PreserveRethPrefixMutator::new(mutator);
    let power: StdPowerMutationalStage<_, _, BytesInput, _, _, _> =
        StdPowerMutationalStage::new(mutator);
    let mut stages = tuple_list!(calibration, power);

    let scheduler = IndexesLenTimeMinimizerScheduler::new(
        &edges_observer,
        GitAwareStdWeightedScheduler::with_schedule(
            &mut state,
            &edges_observer,
            Some(PowerSchedule::fast()),
        ),
    );
    let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

    let mut harness = |input: &BytesInput| {
        let target = input.target_bytes();
        let buf = target.as_slice();
        bench_target(buf);
        ExitKind::Ok
    };

    let mut executor = InProcessExecutor::with_timeout(
        &mut harness,
        tuple_list!(edges_observer, time_observer),
        &mut fuzzer,
        &mut state,
        &mut mgr,
        timeout,
    )?;

    if state.must_load_initial_inputs() {
        state
            .load_initial_inputs(&mut fuzzer, &mut executor, &mut mgr, &[seed_dir.clone()])
            .unwrap_or_else(|_| process::exit(2));
    }

    fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)?;
    Ok(())
}

fn main() {
    let (out_dir, seed_dir, timeout) = parse_args();
    if !Path::new(&seed_dir).is_dir() {
        eprintln!("Seed dir does not exist: {seed_dir:?}");
        process::exit(2);
    }

    if let Err(err) = fuzz(out_dir, &seed_dir, timeout) {
        eprintln!("Fuzzer error: {err:?}");
        process::exit(1);
    }
}
EOF

  cat > "${crate_dir}/src/recent_bug.rs" <<'EOF'
use std::hint::black_box;

#[inline(never)]
pub fn maybe_trigger(data: &[u8], state: u64) {
    if data.len() < 8 || &data[..4] != b"RETH" {
        return;
    }

    let raw = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let mixed = raw
        ^ (state as u32).rotate_left(7)
        ^ ((state >> 32) as u32).rotate_right(3);
    let denom = (mixed & 0x3fff).wrapping_sub(0x211b);

    // Baseline: no crash.
    let x = denom.wrapping_add(1);
    black_box(x);
}
EOF
}

patch_workspace_members() {
  local reth_root="$1"
  python3 - <<'PY'
from pathlib import Path

path = Path("Cargo.toml")
src = path.read_text()

needle = "members = [\n"
ins = '    "crates/libafl-gitaware-bench/",\n'

if ins in src:
    raise SystemExit(0)

idx = src.find(needle)
if idx == -1:
    raise SystemExit("Could not find workspace members list in Cargo.toml")

out = src[: idx + len(needle)] + ins + src[idx + len(needle) :]
path.write_text(out)
PY
}

introduce_recent_bug_commit() {
  local reth_root="$1"
  (
    cd "${reth_root}"
    python3 - <<'PY'
from pathlib import Path

path = Path("crates/libafl-gitaware-bench/src/recent_bug.rs")
src = path.read_text()

needle = "    let x = denom.wrapping_add(1);\n"
replacement = "    let x = 0x1234_5678u32 / denom; // RECENT_BUG\n"

if needle not in src:
    raise SystemExit("Could not find baseline line to replace")

path.write_text(src.replace(needle, replacement, 1))
PY
    git add crates/libafl-gitaware-bench/src/recent_bug.rs
    git commit -qm "bench: introduce RECENT_BUG (recent line)"
  )
}

init_bench_repo() {
  local bench_root="$1"
  local reth_root="${bench_root}/reth"

  rm -rf "${reth_root}"
  mkdir -p "${bench_root}"

  git clone --depth 1 https://github.com/paradigmxyz/reth "${reth_root}" >/dev/null

  local libafl_root
  libafl_root="$(repo_root)"

  write_bench_crate "${reth_root}" "${libafl_root}"
  (cd "${reth_root}" && patch_workspace_members "${reth_root}")

  (
    cd "${reth_root}"
    git config user.name "libafl-git-aware-bench"
    git config user.email "bench@example.invalid"

    git checkout --orphan libafl-bench >/dev/null 2>&1 || true
    git add -A
    GIT_AUTHOR_DATE="2000-01-01T00:00:00Z" \
      GIT_COMMITTER_DATE="2000-01-01T00:00:00Z" \
      git commit -qm "bench: baseline snapshot (old)"
  )

}

build_bench_fuzzer() {
  local bench_root="$1"
  local tools_dir="$2"
  local reth_root="${bench_root}/reth"
  local tc
  tc="$(rust_toolchain)"

  local plugin="${tools_dir}/git-recency-pass.so"
  local stubs_o="${tools_dir}/sancov_stubs.o"

  cat > "${tools_dir}/sancov_stubs.c" <<'EOF'
#include <stdint.h>

__attribute__((weak)) void __sanitizer_cov_trace_pc_guard(uint32_t* guard) {
  (void)guard;
}

__attribute__((weak)) void __sanitizer_cov_trace_pc_guard_init(uint32_t* start, uint32_t* stop) {
  (void)start;
  (void)stop;
}
EOF
  clang-20 -c -O2 "${tools_dir}/sancov_stubs.c" -o "${stubs_o}"

  local rustflags=(
    "-Cpanic=abort"
    "-Cdebuginfo=1"
    "-Clto=off"
    "-Cpasses=sancov-module libafl-git-recency"
    "-Cllvm-args=--sanitizer-coverage-level=3"
    "-Cllvm-args=--sanitizer-coverage-trace-pc-guard"
    "-Zllvm-plugins=${plugin}"
    "-Clink-arg=${stubs_o}"
    "-Clink-arg=-Wl,--no-gc-sections"
  )
  local encoded_rustflags
  encoded_rustflags="$(IFS=$'\x1f'; echo "${rustflags[*]}")"

  (
    cd "${reth_root}"
    LIBAFL_EDGES_MAP_ALLOCATED_SIZE=8388608 \
      CARGO_ENCODED_RUSTFLAGS="${encoded_rustflags}" \
      RUSTC_BOOTSTRAP=1 \
      cargo "+${tc}" build --release -p libafl-gitaware-bench >/dev/null
  )

  echo "${reth_root}/target/release/libafl-gitaware-bench"
}

generate_mapping() {
  local bench_root="$1"
  local tools_dir="$2"
  local fuzzer_bin="$3"

  local mapping_path="${bench_root}/git_recency_map.bin"
  (
    cd "${bench_root}/reth"
    "${tools_dir}/libafl_git_recency_mapgen" --out "${mapping_path}" --bin "${fuzzer_bin}"
  )

  echo "${mapping_path}"
}

write_seeds() {
  local bench_root="$1"
  local in_dir="${bench_root}/in"
  mkdir -p "${in_dir}"

  # One seed that enters the "recent code" path (prefix b"RETH"), but does not crash.
  python3 - <<PY
from pathlib import Path
in_dir = Path(${in_dir@Q})
in_dir.mkdir(parents=True, exist_ok=True)
def prng_bytes(n: int, seed: int) -> bytes:
    x = seed & 0xFFFFFFFF
    out = bytearray()
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0xFFFFFFFF
        out.append((x >> 16) & 0xFF)
    return bytes(out)

(in_dir / "seed_recent_path").write_bytes(b"RETH" + prng_bytes(4096, 0x12345678))
(in_dir / "seed_other").write_bytes(b"NOPE" + prng_bytes(4096, 0x87654321))
PY
}

run_warmup() {
  local bench_root="$1"
  local fuzzer_bin="$2"
  local seed_dir="$3"
  local warmup_secs="$4"

  # Warmup is a short run whose only purpose is to generate a stable-ish starting corpus
  # (the output queue directory). We intentionally:
  #  - run on the baseline snapshot (no RECENT_BUG crash yet)
  #  - disable mapping to avoid git-aware bias during warmup
  #  - fix the rand seed to reduce variance across repeated benchmark runs
  python3 - <<PY
import os
import signal
import subprocess
import time
from pathlib import Path

bench_root = Path(${bench_root@Q})
fuzzer_bin = Path(${fuzzer_bin@Q})
seed_dir = Path(${seed_dir@Q})
warmup_secs = float(${warmup_secs@Q})

out_dir = bench_root / "warmup" / "out"
workdir = bench_root / "warmup" / "workdir"

def _rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            _rm(child)
        path.rmdir()
    else:
        path.unlink()

def _kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()

if out_dir.exists():
    _rm(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
workdir.mkdir(parents=True, exist_ok=True)

cmd = [str(fuzzer_bin), "-o", str(out_dir), "-i", str(seed_dir)]
env = os.environ.copy()
env["LIBAFL_RAND_SEED"] = "0"
env.pop("LIBAFL_GIT_RECENCY_MAPPING_PATH", None)

start = time.time()
proc = subprocess.Popen(
    cmd,
    cwd=str(workdir),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    preexec_fn=os.setsid,
    env=env,
)
try:
    while True:
        if (time.time() - start) >= warmup_secs:
            break
        if proc.poll() is not None:
            break
        time.sleep(0.05)
finally:
    _kill_proc(proc)

queue_dir = out_dir / "queue"
if not queue_dir.is_dir():
    raise SystemExit(f"Warmup corpus missing: {queue_dir}")
PY

  echo "${bench_root}/warmup/out/queue"
}

run_trials() {
  local bench_root="$1"
  local fuzzer_bin="$2"
  local mapping_path="$3"
  local budget_secs="$4"
  local trials="$5"
  local input_dir="$6"

  # Trial runner:
  #  - Runs baseline vs git-aware sequentially (paired) with the same LIBAFL_RAND_SEED.
  #  - Starts each run from the same input corpus directory (the warmup queue).
  #  - Detects a crash by watching for any file to appear in out/crashes/.
  python3 - <<PY
import os
import signal
import math
import statistics
import subprocess
import time
from pathlib import Path

bench_root = Path(${bench_root@Q})
fuzzer_bin = Path(${fuzzer_bin@Q})
mapping_path = Path(${mapping_path@Q})
budget_secs = float(${budget_secs@Q})
trials = int(${trials@Q})
in_dir = Path(${input_dir@Q})

def _rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            _rm(child)
        path.rmdir()
    else:
        path.unlink()

def _kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()

def run_one(variant: str, trial_idx: int, use_mapping: bool) -> float | None:
    out_dir = bench_root / "runs" / variant / f"trial_{trial_idx:03d}" / "out"
    workdir = bench_root / "runs" / variant / f"trial_{trial_idx:03d}" / "workdir"
    if out_dir.exists():
        _rm(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LIBAFL_RAND_SEED"] = str(trial_idx)
    if use_mapping:
        env["LIBAFL_GIT_RECENCY_MAPPING_PATH"] = str(mapping_path)
    else:
        env.pop("LIBAFL_GIT_RECENCY_MAPPING_PATH", None)

    cmd = [str(fuzzer_bin), "-o", str(out_dir), "-i", str(in_dir)]
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
        env=env,
    )

    crash_dir = out_dir / "crashes"
    crash_time = None
    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= budget_secs:
                break

            if crash_dir.is_dir():
                try:
                    if any(p.is_file() for p in crash_dir.iterdir()):
                        crash_time = elapsed
                        break
                except FileNotFoundError:
                    pass

            if proc.poll() is not None:
                break

            time.sleep(0.05)
    finally:
        _kill_proc(proc)

    return crash_time

def run_variant(name: str, use_mapping: bool):
    times = []
    found_flags = []
    for i in range(1, trials + 1):
        t = run_one(name, i, use_mapping)
        if t is None:
            times.append(budget_secs)
            found_flags.append(False)
        else:
            times.append(t)
            found_flags.append(True)
    return times, found_flags

baseline_times, baseline_found_flags = run_variant("baseline", False)
gitaware_times, gitaware_found_flags = run_variant("git-aware", True)

def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    x = (len(values) - 1) * p
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return values[lo]
    frac = x - lo
    return values[lo] + (values[hi] - values[lo]) * frac

def _stats(times: list[float]) -> dict[str, float]:
    return {
        "min": min(times),
        "p25": _pct(times, 0.25),
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "p75": _pct(times, 0.75),
        "max": max(times),
        "std": statistics.pstdev(times) if times else float("nan"),
    }

def _fmt_f64(v: float) -> str:
    if math.isnan(v):
        return "n/a"
    return f"{v:.3f}"

def _render_table(headers: list[str], rows: list[list[str]], right_align_cols: set[int]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    out_lines = []
    header_cells = []
    for i, h in enumerate(headers):
        header_cells.append(h.rjust(widths[i]) if i in right_align_cols else h.ljust(widths[i]))
    out_lines.append("  " + "  ".join(header_cells))

    sep_cells = ["-" * w for w in widths]
    out_lines.append("  " + "  ".join(sep_cells))

    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cells.append(cell.rjust(widths[i]) if i in right_align_cols else cell.ljust(widths[i]))
        out_lines.append("  " + "  ".join(cells))
    return "\n".join(out_lines)

baseline_found = sum(baseline_found_flags)
gitaware_found = sum(gitaware_found_flags)

baseline_s = _stats(baseline_times)
gitaware_s = _stats(gitaware_times)

def _success_pct(found: int) -> float:
    return (100.0 * found / trials) if trials else 0.0

summary_rows = [
    [
        "baseline",
        f"{baseline_found}/{trials}",
        f"{_success_pct(baseline_found):.1f}%",
        _fmt_f64(baseline_s["median"]),
        _fmt_f64(baseline_s["mean"]),
        _fmt_f64(baseline_s["p25"]),
        _fmt_f64(baseline_s["p75"]),
        _fmt_f64(baseline_s["std"]),
        _fmt_f64(baseline_s["min"]),
        _fmt_f64(baseline_s["max"]),
    ],
    [
        "git-aware",
        f"{gitaware_found}/{trials}",
        f"{_success_pct(gitaware_found):.1f}%",
        _fmt_f64(gitaware_s["median"]),
        _fmt_f64(gitaware_s["mean"]),
        _fmt_f64(gitaware_s["p25"]),
        _fmt_f64(gitaware_s["p75"]),
        _fmt_f64(gitaware_s["std"]),
        _fmt_f64(gitaware_s["min"]),
        _fmt_f64(gitaware_s["max"]),
    ],
]

print("")
print("Benchmark results (time-to-first-crash)")
print("")
print(f"  Bench root: {bench_root}")
print(f"  Fuzzer bin: {fuzzer_bin}")
print(f"  Input corpus: {in_dir}")
print(f"  Mapping file: {mapping_path}")
print("")
print("  Definition: wall-clock seconds from fuzzer start until the first crash file appears in")
print("              the output 'crashes' dir.")
print("  Note: if no crash is found within the budget, that trial counts as 'budget' seconds.")
print(f"  Trials: {trials}")
print(f"  Budget: {budget_secs:.2f}s")
print("")

print("Summary (timeouts are capped to budget seconds):")
print(
    _render_table(
        headers=[
            "variant",
            "found",
            "success",
            "median_s",
            "mean_s",
            "p25_s",
            "p75_s",
            "std_s",
            "min_s",
            "max_s",
        ],
        rows=summary_rows,
        right_align_cols=set(range(1, 10)),
    )
)

baseline_med = baseline_s["median"]
gitaware_med = gitaware_s["median"]
delta_med = baseline_med - gitaware_med
speedup = (baseline_med / gitaware_med) if gitaware_med > 0 else float("inf")

wins = ties = losses = 0
for b, g in zip(baseline_times, gitaware_times, strict=True):
    if g < b:
        wins += 1
    elif g > b:
        losses += 1
    else:
        ties += 1

print("")
print("Paired comparison (by seed, lower is better):")
print(f"  Wins/ties/losses (git-aware vs baseline): {wins}/{ties}/{losses}")
print(f"  Median delta (baseline - git-aware): {delta_med:.3f}s")
print(f"  Median speedup factor: {speedup:.3f}x")

if trials <= 50:
    print("")
    print("Per-trial breakdown (capped at budget on timeout):")
    per_rows = []
    for i, (b, b_ok, g, g_ok) in enumerate(
        zip(baseline_times, baseline_found_flags, gitaware_times, gitaware_found_flags, strict=True),
        start=1,
    ):
        b_cell = f"{b:.3f}" if b_ok else "timeout"
        g_cell = f"{g:.3f}" if g_ok else "timeout"
        winner = "git-aware" if g < b else ("baseline" if b < g else "tie")
        per_rows.append([str(i), b_cell, g_cell, winner])

    print(
        _render_table(
            headers=["trial", "baseline_s", "git-aware_s", "winner"],
            rows=per_rows,
            right_align_cols={0, 1, 2},
        )
    )
else:
    print("")
    print(f"Per-trial breakdown omitted (trials={trials}; set <= 50 to print).")
print("")
PY
}

main() {
  local trials="${DEFAULT_TRIALS}"
  local budget_secs="${DEFAULT_BUDGET_SECS}"
  local warmup_secs="${DEFAULT_WARMUP_SECS}"
  local bench_root=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --trials)
        trials="${2:-}"
        shift 2
        ;;
      --budget)
        budget_secs="${2:-}"
        shift 2
        ;;
      --warmup)
        warmup_secs="${2:-}"
        shift 2
        ;;
      --bench-root)
        bench_root="${2:-}"
        shift 2
        ;;
      *)
        echo "Unknown arg: $1" >&2
        usage
        exit 2
        ;;
    esac
  done

  if [[ -z "${bench_root}" ]]; then
    bench_root="${BENCH_ROOT:-}"
  fi
  if [[ -z "${bench_root}" ]]; then
    bench_root="$(mktemp_dir)"
  else
    mkdir -p "${bench_root}"
  fi
  bench_root="$(cd "${bench_root}" && pwd)"

  ensure_host_deps

  local libafl_root
  libafl_root="$(repo_root)"

  local tools_dir="${bench_root}/tools"
  build_libafl_tools "${libafl_root}" "${tools_dir}"

  init_bench_repo "${bench_root}"
  write_seeds "${bench_root}"

  # Baseline snapshot: build once and generate a stable-ish corpus to start from.
  # This warmup run is intentionally done before we introduce the crashing line, so it doesn't
  # terminate early and skew the starting corpus.
  local seed_dir="${bench_root}/in"
  local baseline_bin
  baseline_bin="$(build_bench_fuzzer "${bench_root}" "${tools_dir}")"
  if [[ ! -x "${baseline_bin}" ]]; then
    echo "Baseline fuzzer binary not found: ${baseline_bin}" >&2
    exit 1
  fi
  run_warmup "${bench_root}" "${baseline_bin}" "${seed_dir}" "${warmup_secs}"
  local warm_corpus="${bench_root}/warmup/out/queue"

  # Now introduce the "recent" bug line and rebuild.
  introduce_recent_bug_commit "${bench_root}/reth"

  local fuzzer_bin
  fuzzer_bin="$(build_bench_fuzzer "${bench_root}" "${tools_dir}")"
  if [[ ! -x "${fuzzer_bin}" ]]; then
    echo "Fuzzer binary not found: ${fuzzer_bin}" >&2
    exit 1
  fi

  local mapping_path
  mapping_path="$(generate_mapping "${bench_root}" "${tools_dir}" "${fuzzer_bin}")"
  if [[ ! -f "${mapping_path}" ]]; then
    echo "Mapping file not found: ${mapping_path}" >&2
    exit 1
  fi

  run_trials "${bench_root}" "${fuzzer_bin}" "${mapping_path}" "${budget_secs}" "${trials}" "${warm_corpus}"
}

main "$@"
