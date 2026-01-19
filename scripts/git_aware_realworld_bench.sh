#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/git_aware_realworld_bench.sh [--run] [--trials N] [--budget SECS]
                                      [--baseline-wrappers DIR]
                                      [--gitaware-wrappers DIR]
                                      [--bench-root DIR]

What it does:
  - Creates a small "real-world-ish" benchmark target repo using `stb_image.h`
    (copied from this LibAFL repo) in /tmp (or --bench-root)
  - Sets the baseline commit date to be "old" (so blame timestamps are old)
  - Adds a new commit that introduces a crashing line ("RECENT_BUG")
  - Optionally runs N fuzzing trials and prints median time-to-first-crash
    for baseline vs git-aware.

Notes:
  - The target repo is local-only and intentionally contains a crashing line.
  - For a fair comparison, build baseline and git-aware wrappers from two
    different worktrees/dirs, and pass both wrapper directories.
  - With --run, both --baseline-wrappers and --gitaware-wrappers are required.
  - This script avoids git-based cloning (some environments block it).
EOF
}

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

mktemp_dir() {
  mktemp -d "/tmp/libafl_git_aware_realworld_bench.XXXXXXXX"
}

write_target_repo() {
  local dir="$1"
  local repo_root="$2"

  mkdir -p "${dir}/target_repo"
  cd "${dir}/target_repo"

  git init -q

  git config user.name "libafl-git-aware-bench"
  git config user.email "bench@example.invalid"

  cp "${repo_root}/fuzzers/inprocess/libfuzzer_stb_image/stb_image.h" stb_image.h

  cat > fuzz.cc <<'EOF'
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define STBI_ASSERT(x)
#define STBI_NO_SIMD
#define STBI_NO_LINEAR
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

extern "C" void libafl_recent_bug(const uint8_t *data, size_t size);

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  int x, y, channels;

  if (!stbi_info_from_memory(data, (int)size, &x, &y, &channels)) {
    return 0;
  }

  libafl_recent_bug(data, size);

  /* exit if the image is larger than ~80MB */
  if (y && x > (80000000 / 4) / y) {
    return 0;
  }

  unsigned char *img = stbi_load_from_memory(data, (int)size, &x, &y, &channels, 4);
  free(img);

  return 0;
}
EOF

  cat > libafl_recent_bug.c <<'EOF'
#include <stddef.h>
#include <stdint.h>

__attribute__((noinline))
void libafl_recent_bug(const uint8_t *data, size_t size) {
  if (size < 2) {
    return;
  }

  if (data[size - 1] == 0) {
    return;
  }

  uint16_t v = ((uint16_t)data[size - 2] << 8) | (uint16_t)data[size - 1];
  volatile uint32_t x = (uint32_t)v + 1U;
  (void)x;
}
EOF

  git add stb_image.h fuzz.cc libafl_recent_bug.c

  # Make the whole baseline state "old" so blame timestamps are old.
  GIT_AUTHOR_DATE="2000-01-01T00:00:00Z" \
    GIT_COMMITTER_DATE="2000-01-01T00:00:00Z" \
    git commit -qm "bench: baseline (old commit date)"

  python3 - <<'PY'
from pathlib import Path

path = Path("libafl_recent_bug.c")
src = path.read_text()

src = src.replace(
    "  uint16_t v = ((uint16_t)data[size - 2] << 8) | (uint16_t)data[size - 1];\n",
    "  uint16_t v = (uint16_t)(((uint16_t)data[size - 2] << 8) | (uint16_t)data[size - 1]);\n",
    1,
)
src = src.replace(
    "  volatile uint32_t x = (uint32_t)v + 1U;\n",
    "  volatile uint32_t x = 0x12345678U / (uint32_t)(v - 0x1337U); // RECENT_BUG\n",
    1,
)

path.write_text(src)
PY

  git commit -am "bench: introduce RECENT_BUG (recent line)" -q
}

build_fuzz_target() {
  local wrappers_dir="$1"
  local target_repo="$2"
  local out_bin="$3"
  local tag="$4"
  local mapping_out="$5"

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

    local env_prefix=()
    if [[ "${tag}" == "gitaware" ]]; then
      env_prefix=(env "LIBAFL_GIT_RECENCY_MAPPING_PATH=${mapping_out}")
    fi

    "${env_prefix[@]}" "${wrappers_dir}/libafl_cc" --libafl-no-link -O2 -g -I. -c libafl_recent_bug.c -o "libafl_recent_bug.${tag}.o"
    "${env_prefix[@]}" "${wrappers_dir}/libafl_cxx" --libafl-no-link -O2 -g -I. -c fuzz.cc -o "fuzz.${tag}.o"

    "${env_prefix[@]}" "${wrappers_dir}/libafl_cxx" --libafl \
      "libafl_recent_bug.${tag}.o" "fuzz.${tag}.o" \
      -o "${out_bin}"
  )
}

run_trials() {
  local bench_root="$1"
  local in_dir="$2"
  local budget_secs="$3"
  local trials="$4"
  local baseline_bin="$5"
  local gitaware_bin="$6"
  local mapping_path="$7"

  python3 - <<PY
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

bench_root = Path(${bench_root@Q})
in_dir = Path(${in_dir@Q})
budget_secs = float(${budget_secs@Q})
trials = int(${trials@Q})
baseline_bin = Path(${baseline_bin@Q})
gitaware_bin = Path(${gitaware_bin@Q})
mapping_path = Path(${mapping_path@Q})

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

def run_trial(bin_path: Path, variant: str, trial_idx: int, env_overrides):
    out_dir = bench_root / "runs" / variant / f"trial_{trial_idx:03d}" / "out"
    if out_dir.exists():
        _rm(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workdir = bench_root / "runs" / variant / f"trial_{trial_idx:03d}" / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    _rm(workdir / "libafl_unix_shmem_server")

    cmd = [str(bin_path), "-o", str(out_dir), "-i", str(in_dir)]
    start = time.time()

    env = os.environ.copy()
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v

    popen_kwargs = {}
    if os.name != "nt":
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        env=env,
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
                    pass

            if proc.poll() is not None:
                break

            time.sleep(0.05)
    finally:
        _kill_proc(proc)

    return crash_time

def run_pair(trial_idx: int):
    times = {}

    t = run_trial(
        baseline_bin,
        "baseline",
        trial_idx,
        {"LIBAFL_GIT_RECENCY_MAPPING_PATH": None},
    )
    times["baseline"] = budget_secs if t is None else t

    t = run_trial(
        gitaware_bin,
        "gitaware",
        trial_idx,
        {"LIBAFL_GIT_RECENCY_MAPPING_PATH": str(mapping_path)},
    )
    times["gitaware"] = budget_secs if t is None else t

    return times

baseline_times = []
gitaware_times = []
baseline_found = 0
gitaware_found = 0

for i in range(1, trials + 1):
    pair = run_pair(i)
    bt = pair["baseline"]
    gt = pair["gitaware"]
    baseline_times.append(bt)
    gitaware_times.append(gt)
    if bt < budget_secs:
        baseline_found += 1
    if gt < budget_secs:
        gitaware_found += 1

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
print("  Variant    median_s  found/trials")
print(f"  baseline   {med(baseline_times):7.3f}  {baseline_found}/{trials}")
print(f"  git-aware  {med(gitaware_times):7.3f}  {gitaware_found}/{trials}")
print("")
PY
}

main() {
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

  if [[ -n "${bench_root}" ]]; then
    mkdir -p "${bench_root}"
  else
    bench_root="$(mktemp_dir)"
  fi

  local repo
  repo="$(repo_root)"

  local target_root="${bench_root}/target"
  write_target_repo "${target_root}" "${repo}"

  local target_repo="${target_root}/target_repo"
  local mapping_path="${bench_root}/git_recency_map.bin"

  mkdir -p "${bench_root}/in"
  python3 - <<PY
import os
import struct
os.makedirs("${bench_root}/in", exist_ok=True)

def bmp(width: int, height: int, last_byte: int) -> bytes:
    row_bytes = ((width * 3 + 3) // 4) * 4
    img_size = row_bytes * height
    file_size = 14 + 40 + img_size
    offset = 14 + 40

    header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, offset)
    dib = struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, img_size, 0, 0, 0, 0)

    row = bytearray()
    for _ in range(width):
        row += bytes([0, 0, 0])  # BGR
    row += b"\\x00" * (row_bytes - width * 3)

    data = bytes(row) * height
    data = data[:-1] + bytes([last_byte & 0xFF])
    return header + dib + data

open("${bench_root}/in/seed_bmp_zero", "wb").write(bmp(2, 2, 0))
open("${bench_root}/in/seed_bmp_one", "wb").write(bmp(2, 2, 1))
PY

  if [[ "${run}" -eq 0 ]]; then
    cat <<EOF
Created benchmark target git repo:
  ${target_repo}

It has 2 local commits:
  - baseline: imports stb_image + harness (old commit date)
  - bug: modifies libafl_recent_bug.c to add a crash line (marked 'RECENT_BUG')

Quick checks:
  cd "${target_repo}"
  rg -n "RECENT_BUG" libafl_recent_bug.c
  git blame -L 1,200 libafl_recent_bug.c | head

To run the automated benchmark:
  scripts/git_aware_realworld_bench.sh --run --trials ${trials} --budget ${budget_secs} \\
    --baseline-wrappers "<DIR_WITH_libafl_cc_and_libafl_cxx>" \\
    --gitaware-wrappers "<DIR_WITH_libafl_cc_and_libafl_cxx>"
EOF
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

  build_fuzz_target "${baseline_wrappers}" "${target_repo}" "${baseline_bin}" "baseline" "${mapping_path}"
  build_fuzz_target "${gitaware_wrappers}" "${target_repo}" "${gitaware_bin}" "gitaware" "${mapping_path}"

  run_trials "${bench_root}" "${bench_root}/in" "${budget_secs}" "${trials}" "${baseline_bin}" "${gitaware_bin}" "${mapping_path}"
}

main "$@"
