#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/git_aware_scheduler_bench.sh [--run] [--trials N] [--budget SECS]
                                      [--baseline-wrappers DIR]
                                      [--gitaware-wrappers DIR]
                                      [--bench-root DIR]

What it does:
  - Creates a tiny target git repo in /tmp with 2 commits:
    - baseline: no bug
    - bug: adds a single newly-introduced crashing line (marked 'RECENT_BUG')
  - Optionally runs N fuzzing trials and prints median time-to-first-crash
    for baseline vs git-aware wrappers.

Notes:
  - Never push the generated benchmark repo anywhere.
  - For a fair comparison, build baseline and git-aware wrappers from two
    different worktrees/dirs, and pass both wrapper directories.
  - With --run, both --baseline-wrappers and --gitaware-wrappers are required.
EOF
}

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

mktemp_dir() {
  mktemp -d "/tmp/libafl_git_aware_bench.XXXXXXXX"
}

write_target_repo() {
  local dir="$1"

  mkdir -p "${dir}"
  cd "${dir}"
  git init -q
  git config user.name "libafl-git-aware-bench"
  git config user.email "bench@example.invalid"

  cat > fuzz.c <<'EOF'
#include <stddef.h>
#include <stdint.h>

int parse(const uint8_t *data, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  (void)parse(data, size);
  return 0;
}
EOF

  {
    cat <<'EOF'
#include <stddef.h>
#include <stdint.h>

static inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

int parse(const uint8_t *data, size_t size) {
  uint32_t acc = 0x12345678U;

  // Coverage "noise": a lot of independent branches.
  // The goal is to quickly grow the corpus so that baseline scheduling has
  // many choices, making it harder to focus on a single already-covered path.
EOF

    local i
    for i in $(seq 0 511); do
      local c
      c=$(( (i * 131 + 7) & 255 ))
      printf '  if (size > %d && data[%d] == 0x%02X) { acc ^= mix32(%dU); }\n' "${i}" "${i}" "${c}" "${i}"
    done

    cat <<'EOF'

  // A "normal" parsing-ish path. It is easy to reach, but the crash trigger is
  // data-dependent and does not introduce new coverage by itself.
  if (size >= 3 && data[0] == 'B') {
    uint16_t denom = (uint16_t)((uint16_t)data[1] << 8) | (uint16_t)data[2];
    acc ^= (uint32_t)denom;
  }

  return (int)acc;
}
EOF
  } > parser.c

  git add fuzz.c parser.c
  GIT_AUTHOR_DATE="2000-01-01T00:00:00Z" \
    GIT_COMMITTER_DATE="2000-01-01T00:00:00Z" \
    git commit -qm "bench: baseline (no bug)"

  python3 - <<'PY'
from pathlib import Path

path = Path("parser.c")
src = path.read_text()

needle = "    acc ^= (uint32_t)denom;\n"
replacement = (
    "    acc ^= (uint32_t)denom;\n"
    "    volatile uint32_t x = 0x12345678U / (uint32_t)(denom - 0x1337U); // RECENT_BUG\n"
    "    acc ^= x;\n"
)

if needle not in src:
    raise SystemExit("Could not find insertion point in parser.c")

path.write_text(src.replace(needle, replacement, 1))
PY
  git commit -am "bench: introduce RECENT_BUG (recent line)" -q
}

build_fuzz_target() {
  local wrappers_dir="$1"
  local target_repo="$2"
  local out_bin="$3"
  local tag="$4"

  if [[ ! -x "${wrappers_dir}/libafl_cc" ]]; then
    echo "Missing ${wrappers_dir}/libafl_cc (not executable)"
    return 1
  fi
  if [[ ! -x "${wrappers_dir}/libafl_cxx" ]]; then
    echo "Missing ${wrappers_dir}/libafl_cxx (not executable)"
    return 1
  fi

  (
    cd "${target_repo}"
    "${wrappers_dir}/libafl_cc" --libafl-no-link -O2 -g -c parser.c -o "parser.${tag}.o"
    "${wrappers_dir}/libafl_cc" --libafl-no-link -O2 -g -c fuzz.c -o "fuzz.${tag}.o"
    "${wrappers_dir}/libafl_cxx" --libafl "parser.${tag}.o" "fuzz.${tag}.o" -o "${out_bin}"
  )
}

run_trials() {
  local bench_root="$1"
  local target_repo="$2"
  local in_dir="$3"
  local budget_secs="$4"
  local trials="$5"
  local baseline_bin="$6"
  local gitaware_bin="$7"

  python3 - <<PY
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

bench_root = Path(${bench_root@Q})
target_repo = Path(${target_repo@Q})
in_dir = Path(${in_dir@Q})
budget_secs = float(${budget_secs@Q})
trials = int(${trials@Q})
baseline_bin = Path(${baseline_bin@Q})
gitaware_bin = Path(${gitaware_bin@Q})

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

    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        proc.terminate()

    try:
        proc.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        proc.kill()
    proc.wait()

def run_trial(bin_path: Path, variant: str, trial_idx: int):
    out_dir = bench_root / "runs" / variant / f"trial_{trial_idx:03d}" / "out"
    if out_dir.exists():
        _rm(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep the per-run shmem server path isolated.
    workdir = bench_root / "runs" / variant / f"trial_{trial_idx:03d}" / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    # Avoid stale servers between runs.
    _rm(workdir / "libafl_unix_shmem_server")

    cmd = [str(bin_path), "-o", str(out_dir), "-i", str(in_dir)]
    start = time.time()

    popen_kwargs = {}
    if os.name != "nt":
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **popen_kwargs,
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
                    # directory raced
                    pass

            if proc.poll() is not None:
                break

            time.sleep(0.05)
    finally:
        _kill_proc(proc)

    return crash_time

def run_variant(name: str, bin_path: Path):
    times = []
    found = 0
    for i in range(1, trials + 1):
        t = run_trial(bin_path, name, i)
        if t is None:
            times.append(budget_secs)
        else:
            found += 1
            times.append(t)
    return times, found

baseline_times, baseline_found = run_variant("baseline", baseline_bin)
gitaware_times, gitaware_found = run_variant("gitaware", gitaware_bin)

def med(xs):
    return statistics.median(xs)

print("")
print("Results (time-to-first-crash)")
print("")
print("  Definition: wall-clock seconds from fuzzer start until the first crash file")
print("              appears in the output 'crashes' dir.")
print("  Note: if no crash is found within the budget, that trial counts as 'budget' seconds.")
print(f"  Trials: {trials}")
print(f"  Budget: {budget_secs:.2f}s")
print("")
print("  Table columns:")
print("    - median_s: median time-to-first-crash (seconds; missing == budget)")
print("    - found/trials: number of trials that found a crash within the budget")
print("")
print("  Variant    median_s  found/trials")
print(f"  baseline   {med(baseline_times):7.3f}  {baseline_found}/{trials}")
print(f"  git-aware  {med(gitaware_times):7.3f}  {gitaware_found}/{trials}")
print("")
PY
}

print_next_steps() {
  local repo_root="$1"
  local bench_root="$2"
  local target_repo="$3"

  local fuzzbench_dir="${repo_root}/fuzzers/inprocess/fuzzbench"
  local fuzzbench_target="${fuzzbench_dir}/target/release"

  cat <<EOF
Created benchmark target git repo:
  ${target_repo}

It has 2 local commits:
  - baseline: no bug
  - bug: adds a single recently-introduced crashing line (marked as 'RECENT_BUG')

Quick checks:
  cd "${target_repo}"
  grep -n "RECENT_BUG" parser.c
  git blame -L <line>,<line> parser.c

Build a baseline fuzzer binary (uses fuzzbench's current scheduler):
  cd "${fuzzbench_dir}"
  cargo build --profile=release

  cd "${target_repo}"
  "${fuzzbench_target}/libafl_cc" --libafl-no-link -O2 -g -c parser.c -o parser.o
  "${fuzzbench_target}/libafl_cc" --libafl-no-link -O2 -g -c fuzz.c -o fuzz.o
  "${fuzzbench_target}/libafl_cxx" --libafl parser.o fuzz.o -o fuzz_target

Run it:
  mkdir -p "${bench_root}/in" "${bench_root}/out"
  python3 - <<'PY'
import os
os.makedirs("${bench_root}/in", exist_ok=True)
open("${bench_root}/in/seed_a", "wb").write(b"A" * 512)
open("${bench_root}/in/seed_b", "wb").write(b"B" + b"\\x00\\x00" + b"C" * 509)
PY

  # Run for ~30s (Linux: timeout 30s ...; macOS with coreutils: gtimeout 30s ...),
  # or just run and Ctrl+C.
  ./fuzz_target -o "${bench_root}/out" -i "${bench_root}/in"

To compare against the git-aware scheduler:
  - Build an equivalent fuzzer binary with the new git-aware scheduler enabled
  - Re-run the same seeds/time budget and compare time-to-first-crash
EOF
}

main() {
  local repo
  repo="$(repo_root)"

  local run=0
  local trials=10
  local budget_secs=30
  local baseline_wrappers=""
  local gitaware_wrappers=""
  local bench_root=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --run)
        run=1
        shift
        ;;
      --trials)
        trials="${2:-}"
        shift 2
        ;;
      --budget)
        budget_secs="${2:-}"
        shift 2
        ;;
      --baseline-wrappers)
        baseline_wrappers="${2:-}"
        shift 2
        ;;
      --gitaware-wrappers)
        gitaware_wrappers="${2:-}"
        shift 2
        ;;
      --bench-root)
        bench_root="${2:-}"
        shift 2
        ;;
      *)
        echo "Unknown arg: $1"
        echo ""
        usage
        exit 2
        ;;
    esac
  done

  if [[ -z "${bench_root}" ]]; then
    bench_root="${BENCH_ROOT:-}"
  fi

  if [[ -n "${bench_root}" ]]; then
    mkdir -p "${bench_root}"
  else
    bench_root="$(mktemp_dir)"
  fi

  local target_repo="${bench_root}/target_repo"

  write_target_repo "${target_repo}"

  mkdir -p "${bench_root}/in"
  python3 - <<PY
import os
os.makedirs("${bench_root}/in", exist_ok=True)
open("${bench_root}/in/seed_a", "wb").write(b"A" * 512)
open("${bench_root}/in/seed_b", "wb").write(b"B" + b"\\x00\\x00" + b"C" * 509)
PY

  if [[ "${run}" -eq 0 ]]; then
    print_next_steps "${repo}" "${bench_root}" "${target_repo}"
    echo ""
    echo "To run the automated benchmark:"
    echo "  scripts/git_aware_scheduler_bench.sh --run --trials ${trials} --budget ${budget_secs} \\"
    echo "    --baseline-wrappers \"<DIR_WITH_libafl_cc_and_libafl_cxx>\" \\"
    echo "    --gitaware-wrappers \"<DIR_WITH_libafl_cc_and_libafl_cxx>\""
    return 0
  fi

  if [[ -z "${baseline_wrappers}" ]]; then
    echo "--baseline-wrappers is required with --run"
    exit 2
  fi

  if [[ -z "${gitaware_wrappers}" ]]; then
    echo "--gitaware-wrappers is required with --run"
    exit 2
  fi

  local baseline_bin="${bench_root}/fuzz_target_baseline"
  local gitaware_bin="${bench_root}/fuzz_target_gitaware"

  build_fuzz_target "${baseline_wrappers}" "${target_repo}" "${baseline_bin}" "baseline"
  build_fuzz_target "${gitaware_wrappers}" "${target_repo}" "${gitaware_bin}" "gitaware"

  run_trials "${bench_root}" "${target_repo}" "${bench_root}/in" "${budget_secs}" "${trials}" "${baseline_bin}" "${gitaware_bin}"
}

main "$@"
