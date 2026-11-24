// Research-only blanket fast math. Do not use for correctness-sensitive builds.
#include "llvm/IR/FMF.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class BlanketFastMath : public PassInfoMixin<BlanketFastMath> {
  const char *Stage;
public:
  explicit BlanketFastMath(const char *Stage) : Stage(Stage) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    uint64_t Eligible = 0, Changed = 0, Functions = 0, DeviceFunctions = 0;
    for (Function &F : M) {
      bool HasFP = false;
      for (Instruction &I : instructions(F)) {
        if (!isa<FPMathOperator>(I))
          continue;
        ++Eligible;
        HasFP = true;
        if (!I.getFastMathFlags().isFast()) {
          I.setFastMathFlags(FastMathFlags::getFast());
          ++Changed;
        }
      }
      if (HasFP) {
        ++Functions;
        if (F.getName().contains("ngspice_rs") && F.getName().contains("devices"))
          ++DeviceFunctions;
      }
    }
    errs() << "CIRCUIT_FAST_MATH stage=" << Stage
           << " module=" << M.getModuleIdentifier()
           << " eligible=" << Eligible << " changed=" << Changed
           << " functions=" << Functions << " device_functions=" << DeviceFunctions << "\n";
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "circuit-blanket-fast-math", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                  MPM.addPass(BlanketFastMath("prelink"));
                });
            PB.registerFullLinkTimeOptimizationEarlyEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                  MPM.addPass(BlanketFastMath("full-lto-early"));
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name != "blanket-fast-math") return false;
                  MPM.addPass(BlanketFastMath("explicit"));
                  return true;
                });
          }};
}
