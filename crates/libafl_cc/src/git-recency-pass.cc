/*
   LibAFL - Git recency mapping LLVM pass
   --------------------------------------------------

   This pass records a per-object mapping from SanitizerCoverage pc-guard indices
   to source locations (file + line). The final mapping to `git blame` timestamps
   is produced at link time by `libafl_cc`.

   The mapping is emitted both as a sidecar file (v1) and embedded into the object
   in a dedicated section (v2) so link-time merging can handle static archives.
*/

#include "common-llvm.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

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

static void append_u32_le(std::vector<uint8_t> &out, uint32_t v) {
  out.push_back((uint8_t)(v & 0xff));
  out.push_back((uint8_t)((v >> 8) & 0xff));
  out.push_back((uint8_t)((v >> 16) & 0xff));
  out.push_back((uint8_t)((v >> 24) & 0xff));
}

static void append_u64_le(std::vector<uint8_t> &out, uint64_t v) {
  out.push_back((uint8_t)(v & 0xff));
  out.push_back((uint8_t)((v >> 8) & 0xff));
  out.push_back((uint8_t)((v >> 16) & 0xff));
  out.push_back((uint8_t)((v >> 24) & 0xff));
  out.push_back((uint8_t)((v >> 32) & 0xff));
  out.push_back((uint8_t)((v >> 40) & 0xff));
  out.push_back((uint8_t)((v >> 48) & 0xff));
  out.push_back((uint8_t)((v >> 56) & 0xff));
}

static bool is_sancov_trace_function(StringRef name) {
  return name == "__sanitizer_cov_trace_pc_guard" ||
         name == "__libafl_targets_trace_pc_guard";
}

static bool is_sancov_init_function(StringRef name) {
  return name == "__sanitizer_cov_trace_pc_guard_init";
}

static const Function *called_function_stripped(const CallBase *CB) {
  if (!CB) { return nullptr; }
  Value *V = CB->getCalledOperand();
  if (!V) { return nullptr; }
  V = V->stripPointerCasts();
  return dyn_cast<Function>(V);
}

static DebugLoc find_last_non_instrumentation_debugloc(const BasicBlock &BB) {
  DebugLoc last;
  for (const auto &I : BB) {
    if (isa<DbgInfoIntrinsic>(&I)) { continue; }

    if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (auto *Callee = called_function_stripped(CB)) {
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

    if (I.isTerminator()) { continue; }

    DebugLoc DL = I.getDebugLoc();
    if (DL) { last = DL; }
  }

  return last;
}

class GitRecencyPass : public PassInfoMixin<GitRecencyPass> {
 public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    if (SidecarPath.empty()) { return PreservedAnalyses::all(); }
    // Collect source locations for each *instrumented* basic block.
    //
    // We only record blocks that contain a trace-pc-guard hook call. This keeps
    // the emitted entry count aligned with the number of guards in the object.
    std::vector<LocEntry> ordered;
    for (auto &F : M) {
      if (isIgnoreFunction(&F)) { continue; }
      for (auto &BB : F) {
        bool has_trace_pc_guard = false;
        for (auto const &I : BB) {
          auto *CB = dyn_cast<CallBase>(&I);
          if (!CB) { continue; }
          if (auto *Callee = called_function_stripped(CB)) {
            if (is_sancov_trace_function(Callee->getName())) {
              has_trace_pc_guard = true;
              break;
            }
          }
        }
        if (!has_trace_pc_guard) { continue; }

        DebugLoc DL = find_last_non_instrumentation_debugloc(BB);

        LocEntry E;
        if (DL) {
          const auto *Loc = DL.get();
          if (Loc) {
            if (auto *File = Loc->getFile()) {
              std::string dir = File->getDirectory().str();
              std::string fname = File->getFilename().str();
              if (!dir.empty()) {
                E.path = dir + "/" + fname;
              } else {
                E.path = fname;
              }
              E.line = Loc->getLine();
              E.known = (!E.path.empty() && E.line != 0);
            }
          }
        }

        ordered.push_back(std::move(E));
      }
    }

    std::vector<uint8_t> blob;
    blob.insert(blob.end(), kMagic, kMagic + sizeof(kMagic));
    append_u64_le(blob, static_cast<uint64_t>(ordered.size()));

    for (auto const &E : ordered) {
      if (!E.known) {
        append_u32_le(blob, 0);
        append_u32_le(blob, 0);
        continue;
      }

      append_u32_le(blob, E.line);
      append_u32_le(blob, static_cast<uint32_t>(E.path.size()));
      blob.insert(blob.end(), E.path.begin(), E.path.end());
    }

    // v1: sidecar file
    std::ofstream out(SidecarPath, std::ios::binary | std::ios::out);
    if (!out.is_open()) {
      FATAL("Could not open git recency sidecar for writing: %s\n",
            SidecarPath.c_str());
    }
    out.write(reinterpret_cast<const char *>(blob.data()),
              static_cast<std::streamsize>(blob.size()));
    out.close();

    // v2: embedded section (supports static archives at link time)
    auto &Ctx = M.getContext();
    ArrayType *arrayTy = ArrayType::get(IntegerType::get(Ctx, 8), blob.size());
    GlobalVariable *meta = new GlobalVariable(
        M, arrayTy, true, GlobalVariable::PrivateLinkage,
        ConstantDataArray::get(Ctx, ArrayRef<uint8_t>(blob.data(), blob.size())),
        "libafl_gitrecency_" + M.getName());
    meta->setAlignment(Align(1));
#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__) || \
    defined(__OpenBSD__) || defined(__DragonFly__)
    meta->setSection("libafl_gitrec");
#elif defined(__APPLE__)
    meta->setSection("__DATA,__libafl_gitrec");
#endif
    GlobalValue *used[] = {meta};
    appendToCompilerUsed(M, used);

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
