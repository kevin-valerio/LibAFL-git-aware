/*
   LibAFL - Git recency mapping LLVM pass
   --------------------------------------------------

   This pass records a per-object mapping from SanitizerCoverage pc-guard indices
   to source locations (file + line). The final mapping to `git blame` timestamps
   is produced at link time by `libafl_cc`.

   v1 scope: only direct `.o` inputs are supported when merging at link time.
*/

#include "common-llvm.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Instructions.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace llvm;

static cl::opt<std::string> SidecarPath(
    "libafl-git-recency-sidecar",
    cl::desc("Write per-object git-recency sidecar metadata to this path"),
    cl::init(std::string("")), cl::NotHidden);

namespace {

static constexpr const char kMagic[8] = {'L', 'A', 'F', 'L',
                                         'G', 'I', 'T', '1'};

struct LocEntry {
  std::string path;
  uint32_t    line = 0;
  bool        known = false;
};

static void write_u32_le(std::ofstream &out, uint32_t v) {
  uint8_t b[4];
  b[0] = (uint8_t)(v & 0xff);
  b[1] = (uint8_t)((v >> 8) & 0xff);
  b[2] = (uint8_t)((v >> 16) & 0xff);
  b[3] = (uint8_t)((v >> 24) & 0xff);
  out.write(reinterpret_cast<const char *>(b), sizeof(b));
}

static void write_u64_le(std::ofstream &out, uint64_t v) {
  uint8_t b[8];
  b[0] = (uint8_t)(v & 0xff);
  b[1] = (uint8_t)((v >> 8) & 0xff);
  b[2] = (uint8_t)((v >> 16) & 0xff);
  b[3] = (uint8_t)((v >> 24) & 0xff);
  b[4] = (uint8_t)((v >> 32) & 0xff);
  b[5] = (uint8_t)((v >> 40) & 0xff);
  b[6] = (uint8_t)((v >> 48) & 0xff);
  b[7] = (uint8_t)((v >> 56) & 0xff);
  out.write(reinterpret_cast<const char *>(b), sizeof(b));
}

static bool is_sancov_trace_function(StringRef name) {
  return name == "__sanitizer_cov_trace_pc_guard" ||
         name == "__libafl_targets_trace_pc_guard";
}

static bool is_sancov_init_function(StringRef name) {
  return name == "__sanitizer_cov_trace_pc_guard_init";
}

static bool resolve_guard_ptr(Value *V, const GlobalVariable *&out_gv,
                              uint64_t &out_idx) {
  if (!V) { return false; }

  Value *stripped = V->stripPointerCasts();
  if (auto *GEP = dyn_cast<GEPOperator>(stripped)) {
    Value *base = GEP->getPointerOperand()->stripPointerCasts();
    auto  *GV = dyn_cast<GlobalVariable>(base);
    if (!GV) { return false; }

    uint64_t idx = 0;
    for (auto I = GEP->idx_begin(); I != GEP->idx_end(); ++I) {
      auto *CI = dyn_cast<ConstantInt>(I->get());
      if (!CI) { return false; }
      idx = CI->getZExtValue();
    }

    out_gv = GV;
    out_idx = idx;
    return true;
  }

  if (auto *GV = dyn_cast<GlobalVariable>(stripped)) {
    out_gv = GV;
    out_idx = 0;
    return true;
  }

  return false;
}

static DebugLoc find_first_non_instrumentation_debugloc(const BasicBlock &BB) {
  for (const auto &I : BB) {
    if (isa<DbgInfoIntrinsic>(&I)) { continue; }

    if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (auto *Callee = CB->getCalledFunction()) {
        auto name = Callee->getName();
        if (is_sancov_trace_function(name) || is_sancov_init_function(name)) {
          continue;
        }
        // Skip other sanitizer/afl-style instrumentation helpers.
#if LLVM_VERSION_MAJOR >= 18
        if (name.starts_with("__sanitizer_cov") || name.starts_with("llvm.") ||
            name.starts_with("__afl") || name.starts_with("__sancov")) {
#else
        if (name.startswith("__sanitizer_cov") || name.startswith("llvm.") ||
            name.startswith("__afl") || name.startswith("__sancov")) {
#endif
          continue;
        }
      }
    }

    DebugLoc DL = I.getDebugLoc();
    if (DL) { return DL; }
  }

  return DebugLoc();
}

static uint64_t guard_global_len(const GlobalVariable *GV) {
  if (!GV) { return 0; }
  Type *VT = GV->getValueType();
  if (auto *AT = dyn_cast<ArrayType>(VT)) { return AT->getNumElements(); }
  return 1;
}

class GitRecencyPass : public PassInfoMixin<GitRecencyPass> {
 public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    if (SidecarPath.empty()) { return PreservedAnalyses::all(); }

    std::unordered_map<const GlobalVariable *, std::vector<LocEntry>> entries;
    std::vector<const GlobalVariable *>                             guard_order;

    // Prefer the runtime init call order (matches the order in which this
    // module's guards get indices assigned).
    for (auto &F : M) {
      for (auto &BB : F) {
        for (auto &I : BB) {
          auto *CB = dyn_cast<CallBase>(&I);
          if (!CB) { continue; }
          auto *Callee = CB->getCalledFunction();
          if (!Callee) { continue; }
          if (!is_sancov_init_function(Callee->getName())) { continue; }
          if (CB->arg_size() < 2) { continue; }

          const GlobalVariable *GV = nullptr;
          uint64_t              idx = 0;
          if (!resolve_guard_ptr(CB->getArgOperand(0), GV, idx)) { continue; }

          if (std::find(guard_order.begin(), guard_order.end(), GV) ==
              guard_order.end()) {
            guard_order.push_back(GV);
          }

          if (entries.find(GV) == entries.end()) {
            uint64_t len = guard_global_len(GV);
            entries.emplace(GV, std::vector<LocEntry>(static_cast<size_t>(len)));
          }
        }
      }
    }

    // Collect source locations for each trace site.
    for (auto &F : M) {
      if (isIgnoreFunction(&F)) { continue; }
      for (auto &BB : F) {
        DebugLoc DL = find_first_non_instrumentation_debugloc(BB);

        std::string file_path;
        uint32_t    line = 0;
        if (DL) {
          const auto *Loc = DL.get();
          if (Loc) {
            if (auto *File = Loc->getFile()) {
              std::string dir = File->getDirectory().str();
              std::string fname = File->getFilename().str();
              if (!dir.empty()) {
                file_path = dir + "/" + fname;
              } else {
                file_path = fname;
              }
              line = Loc->getLine();
            }
          }
        }

        for (auto &I : BB) {
          auto *CB = dyn_cast<CallBase>(&I);
          if (!CB) { continue; }
          auto *Callee = CB->getCalledFunction();
          if (!Callee) { continue; }

          if (!is_sancov_trace_function(Callee->getName())) { continue; }
          if (CB->arg_size() < 1) { continue; }

          const GlobalVariable *GV = nullptr;
          uint64_t              guard_idx = 0;
          if (!resolve_guard_ptr(CB->getArgOperand(0), GV, guard_idx)) {
            continue;
          }

          if (entries.find(GV) == entries.end()) {
            uint64_t len = guard_global_len(GV);
            entries.emplace(GV, std::vector<LocEntry>(static_cast<size_t>(len)));
          }
          auto &vec = entries[GV];
          if (guard_idx >= vec.size()) { continue; }

          auto &E = vec[static_cast<size_t>(guard_idx)];
          if (!file_path.empty() && line != 0) {
            E.path = file_path;
            E.line = line;
            E.known = true;
          }
        }
      }
    }

    if (guard_order.empty()) {
      guard_order.reserve(entries.size());
      for (auto const &KV : entries) {
        guard_order.push_back(KV.first);
      }
      std::sort(guard_order.begin(), guard_order.end(),
                [](const GlobalVariable *A, const GlobalVariable *B) {
                  return A->getName() < B->getName();
                });
    }

    uint64_t total_len = 0;
    for (auto *GV : guard_order) {
      auto it = entries.find(GV);
      if (it != entries.end()) { total_len += it->second.size(); }
    }

    std::ofstream out(SidecarPath, std::ios::binary | std::ios::out);
    if (!out.is_open()) {
      FATAL("Could not open git recency sidecar for writing: %s\n",
            SidecarPath.c_str());
    }

    out.write(kMagic, sizeof(kMagic));
    write_u64_le(out, total_len);

    for (auto *GV : guard_order) {
      auto it = entries.find(GV);
      if (it == entries.end()) { continue; }

      for (auto const &E : it->second) {
        if (!E.known) {
          write_u32_le(out, 0);
          write_u32_le(out, 0);
          continue;
        }

        write_u32_le(out, E.line);
        write_u32_le(out, static_cast<uint32_t>(E.path.size()));
        out.write(E.path.data(), static_cast<std::streamsize>(E.path.size()));
      }
    }

    out.close();
    return PreservedAnalyses::all();
  }
};

}  // namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "GitRecencyPass", "v0.1",
          [](PassBuilder &PB) {
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel OL
#if LLVM_VERSION_MAJOR >= 20
                   ,
                   ThinOrFullLTOPhase Phase
#endif

                ) { MPM.addPass(GitRecencyPass()); });
          }};
}

