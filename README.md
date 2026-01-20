# LibAFL-git-aware

This is a fork of [`AFLplusplus/LibAFL`](https://github.com/AFLplusplus/LibAFL).

The main goal is simple: keep coverage-guided fuzzing, but also prefer testcases that run code changed recently (based on `git blame`).

## What this adds

It adds an opt-in scheduler score that boosts testcases which cover recently changed lines.
“Recent” means the last commit time from `git blame` (`%ct`, epoch seconds).

## How it works

At build time, `libafl_cc` creates a mapping from SanitizerCoverage `trace-pc-guard` map indexes to `git blame` timestamps.
At runtime, the fuzzer loads that mapping and the scheduler uses it to bias corpus selection.

The mapping file format is:

`u64 head_time` + `u64 len` + `len * u64 entries` (all little-endian).

## Implementation details (how it is built)

### Build step 1: record “index → source location” per object file

When you compile with `libafl_cc` and you set `LIBAFL_GIT_RECENCY_MAPPING_PATH`, the wrapper loads an LLVM pass while compiling each object file.

That pass records one source location (`file:line`) per instrumented basic block, in the same order LLVM’s SanitizerCoverage pc-guard pass assigns indices (module function/basic-block iteration order).
To avoid blaming sanitizer/instrumentation code, it uses a non-instrumentation instruction’s debug location in the same basic block (currently: the last non-terminator instruction with a valid debug location, skipping known sanitizer/afl helper calls).

It writes a small sidecar file next to the object file:

`<object>.libafl_git_recency`

This sidecar is a list in pc-guard order for that object:
each entry is either `file:line` or “unknown”.

### Build step 2: merge objects at link time, then run `git blame`

At the final link step, the `libafl_cc` wrapper reads all those sidecar files (for each `.o` on the link line), merges them in the same order, and turns `file:line` into a timestamp using `git blame --line-porcelain`.

We also store the `HEAD` commit time at build time in the file (`head_time`).
This makes the “age” computation stable for a given build.

If a location cannot be mapped (no debug info, file not in the git repo, blame fails), it is treated as “old” and gets timestamp `0`.

To avoid blaming system headers or external code, the link step only blames files that are inside the current git repo root.

### Runtime: how the scheduler uses the mapping

At fuzzer startup, you load the mapping file into `GitRecencyMapMetadata`.

Then `GitRecencyTestcaseScore` wraps the normal weighted score (`CorpusWeightTestcaseScore`) and applies a boost.
For each testcase, it looks at the list of covered map indexes stored in `MapIndexesMetadata` (this is why you must call `.track_indices()` on the map observer).

It computes a cached value:

`tc_time = max(entries[idx])` for all covered `idx`

Then it boosts the testcase weight like this:

`final = base * (1 + alpha * decay)`

The decay is exponential with a fixed half-life of 14 days:

`decay = 2^(-(head_time - tc_time) / half_life)`

`alpha` controls how strong the bias is (default is `2.0`).

To keep runtime overhead low, `tc_time` is computed once per testcase and cached inside `GitRecencyTestcaseMetadata`.

### “How do we know the runtime map index matches the .bin entry?”

This only works because the mapping uses the same index that the target uses at runtime.

With SanitizerCoverage `trace-pc-guard`, each instrumented site has a “guard” variable.
At startup, `__sanitizer_cov_trace_pc_guard_init` assigns each guard a unique number (0, 1, 2, …).
`libafl_targets` then uses that number directly as the index into the coverage map.

So the “basic block ID” you see in `MapIndexesMetadata` is the same number that was written into the guard, and the mapping file stores `entries[that_number]`.

The tricky part is ordering.
The runtime assigns those numbers by walking each object’s guard array, and doing that for all objects.
This implementation makes the build-time merge match that order by writing the per-object metadata in the same order SanitizerCoverage instruments basic blocks in that object (matching the guard array order), embedding it in a dedicated section, and then reading/concatenating that section from the final linked output.

Instrumented `.a` archives are supported as long as they were built with the `libafl_cc` wrappers so that the embedded metadata section is present (archive members pulled in by the linker contribute both `__sancov_guards` and the metadata section to the final binary).
Uninstrumented `.a` archives are ignored for mapping purposes.
Also, this expects plain pc-guard indexing. If you enable instrumentation modes that transform the index (like n-gram or ctx modes), the map index is no longer the raw guard value, and the mapping will not line up.

## How to use it

### 1) Build your target and generate the mapping

Set `LIBAFL_GIT_RECENCY_MAPPING_PATH` to where you want the mapping file.
Then build your target using a `libafl_cc` wrapper so the mapping can be created at the final link step.

Example:

```sh
cargo build --release -p forkserver_libafl_cc --bin libafl_cc
cargo build --release -p forkserver_libafl_cc --bin libafl_cxx

export CC="$(pwd)/target/release/libafl_cc"
export CXX="$(pwd)/target/release/libafl_cxx"
export LIBAFL_GIT_RECENCY_MAPPING_PATH="$(pwd)/git_recency_map.bin"

make CC="$CC" CXX="$CXX" -j"$(nproc)"
```

Notes: you must use `-fsanitize-coverage=trace-pc-guard` and have debug info (`-g`) so coverage sites can be mapped to `file:line`.
Instrumented `.a` archives are supported for mapping generation, as long as they were built with the `libafl_cc` wrappers so the embedded metadata is available in the final linked output.

### 2) Use the git-aware scheduler

Enable index tracking on your map observer (`.track_indices()`), load the mapping file into state, then use the git-aware weighted scheduler.

Minimal sketch (exact types vary by fuzzer):

```rust
use libafl::{
    HasMetadata,
    observers::{CanTrack, StdMapObserver},
    schedulers::{GitAwareStdWeightedScheduler, GitRecencyConfigMetadata, GitRecencyMapMetadata},
};

let edges_observer = StdMapObserver::owned("edges", vec![0u8; 65536]).track_indices();

state.add_metadata(GitRecencyMapMetadata::load_from_file("git_recency_map.bin")?);
state.add_metadata(GitRecencyConfigMetadata::new(2.0)); // optional, default is 2.0

let scheduler = GitAwareStdWeightedScheduler::new(&mut state, &edges_observer);
```

For the full plan, see `docs/plans/git-aware-recent-coverage-scheduler.md`.

## Benchmark

You can run the `reth` benchmark with:

```sh
bash scripts/git_aware_reth_bench.sh --trials 3 --budget 120 --warmup 30
```


This benchmark creates a temporary `reth` git checkout under `/tmp`, adds a small LibAFL in-process fuzzer crate and a freshly committed crashing line marked `RECENT_BUG` so `git blame` treats it as recently changed code, builds the target with SanitizerCoverage (`trace-pc-guard`) plus the git-recency LLVM pass to generate a `pcguard_index -> git blame timestamp` mapping, then runs paired baseline vs git-aware trials and reports the median time-to-first-crash.

### Benchmark 1

Time to find the introduced bug

| trial | baseline_s | git-aware_s | winner   |
|-------|------------|------------|----------|
| 1     | 107.443    | 52.267     | git-aware |
| 2     | 86.076     | 60.382     | git-aware |
| 3     | 31.772     | 124.470    | baseline |
| 4     | 44.089     | 13.072     | git-aware |
| 5     | 211.480    | 27.938     | git-aware |

## License

Same license as upstream `LibAFL`: MIT or Apache-2.0 (see `LICENSE-MIT` and `LICENSE-APACHE`).
