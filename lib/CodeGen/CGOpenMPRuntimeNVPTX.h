//===----- CGOpenMPRuntimeNVPTX.h - Interface to OpenMP NVPTX Runtimes ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation specialized to NVPTX
// targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H

#include "CGOpenMPRuntime.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/IR/CallSite.h"

namespace clang {
namespace CodeGen {

class CGOpenMPRuntimeNVPTX : public CGOpenMPRuntime {
private:
  /// \brief Get the GPU warp size.
  llvm::Function *GetNVPTXWarpSize() {
    return llvm::Intrinsic::getDeclaration(
        &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize);
  }

  /// \brief Get the id of the current thread on the GPU.
  llvm::Function *GetNVPTXThreadID() {
    return llvm::Intrinsic::getDeclaration(
        &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
  }

  /// \brief Get the id of the current block on the GPU.
  llvm::Function *GetNVPTXBlockID() {
    return llvm::Intrinsic::getDeclaration(
        &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  }

  // \brief Get the maximum number of threads in a block of the GPU.
  llvm::Function *GetNVPTXNumThreads() {
    return llvm::Intrinsic::getDeclaration(
        &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  }

  /// \brief Get barrier #n to synchronize all threads in a block.
  llvm::Function *GetNVPTXCTABarrier() {
    return llvm::Intrinsic::getDeclaration(&CGM.getModule(),
                                           llvm::Intrinsic::nvvm_barrier_n);
  }

  /// \brief Get barrier #n to synchronize selected (multiple of 32) threads in
  /// a block.
  llvm::Function *GetNVPTXBarrier() {
    return llvm::Intrinsic::getDeclaration(&CGM.getModule(),
                                           llvm::Intrinsic::nvvm_barrier);
  }

  /// \brief Get the thread id of the OMP master thread.
  /// The master thread id is the first thread (lane) of the last warp in the
  /// GPU block.
  /// Thread id is 0 indexed.
  /// E.g: If NumThreads is 33, master id is 32.
  ///      If NumThreads is 64, master id is 32.
  ///      If NumThreads is 1024, master id is 992.
  llvm::Value *GetMasterThreadID(CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    llvm::Value *NumThreads = Bld.CreateCall(GetNVPTXNumThreads(), {});

    // We assume that the warp size is a multiple of 2.
    llvm::Value *Mask =
        Bld.CreateSub(Bld.CreateCall(GetNVPTXWarpSize(), {}), Bld.getInt32(1));

    return Bld.CreateAnd(Bld.CreateSub(NumThreads, Bld.getInt32(1)),
                         Bld.CreateNot(Mask), "master_tid");
  }

  /// \brief Get number of OMP workers for parallel region after subtracting
  /// the master warp.
  llvm::Value *GetNumWorkers(CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    return Bld.CreateAdd(GetMasterThreadID(CGF), Bld.getInt32(0),
                         "num_workers");
  }

  /// \brief Get thread id in team.
  /// FIXME: Requires an expensive remainder operation.
  llvm::Value *GetTeamThreadId(CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    return Bld.CreateURem(Bld.CreateCall(GetNVPTXThreadID(), {}),
                          GetMasterThreadID(CGF), "team_tid");
  }

  /// \brief Get global thread id.
  llvm::Value *GetGlobalThreadId(CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    return Bld.CreateAdd(Bld.CreateMul(Bld.CreateCall(GetNVPTXBlockID(), {}),
                                       GetNumWorkers(CGF)),
                         GetTeamThreadId(CGF), "global_tid");
  }

  // \brief Synchronize all GPU threads in a block.
  void SyncCTAThreads(CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    llvm::Value *Args[] = {Bld.getInt32(CTA_BARRIER)};
    Bld.CreateCall(GetNVPTXCTABarrier(), Args);
  }

public:
  explicit CGOpenMPRuntimeNVPTX(CodeGenModule &CGM);

  /// \brief Complete processing for this module.  Called once per module,
  /// after all targets are processed.
  void release() override;

  /// \brief Initialize master-worker control state.
  void initializeEnvironment();

  /// \brief Finalize master-worker control state after all targets are
  /// processed.
  void finalizeEnvironment();

  /// \brief Emit outlined function for 'target' directive on the NVPTX
  /// device.
  /// \param D Directive to emit.
  /// \param ParentName Name of the function that encloses the target region.
  /// \param OutlinedFn Outlined function value to be defined by this call.
  /// \param OutlinedFnID Outlined function ID value to be defined by this call.
  /// \param IsOffloadEntry True if the outlined function is an offload entry.
  /// An outlined function may not be an entry if, e.g. the if clause always
  /// evaluates to false.
  virtual void emitTargetOutlinedFunction(const OMPExecutableDirective &D,
                                          StringRef ParentName,
                                          llvm::Function *&OutlinedFn,
                                          llvm::Constant *&OutlinedFnID,
                                          bool IsOffloadEntry) override;

private:
  // Named barriers for synchronization across subsets of CUDA threads.
  enum BARRIERS {
    // Synchronize all threads in CTA (master + all workers).
    CTA_BARRIER = 0,
    // Synchronize all active worker threads at L1 parallelism.
    L1_BARRIER = 1
  };

  // NVPTX Address space
  enum ADDRESS_SPACE { GLOBAL_ADDRESS_SPACE = 1, SHARED_ADDRESS_SPACE = 3 };

  // Master-worker control state.
  llvm::GlobalVariable *ActiveWorkers;
  llvm::GlobalVariable *WorkID;
  llvm::GlobalVariable *WorkArgs;
  // Maximum number of captured variables sent to any work function in
  // this compilation unit.
  size_t MaxWorkArgs;

  // Pointers to outlined function work for workers.
  llvm::SmallVector<llvm::Function *, 16> Work;

  class EntryFunctionState {
  public:
    const StringRef ParentName;
    const StringRef EntryFnName;
    const unsigned DeviceID;
    const unsigned FileID;
    const unsigned Line;
    const unsigned Column;
    llvm::Function *OutlinedFn;
    llvm::Constant *OutlinedFnID;
    const CapturedStmt &CS;
    llvm::BasicBlock *ExitBB;

    EntryFunctionState(StringRef ParentName, StringRef EntryFnName,
                       unsigned DeviceID, unsigned FileID, unsigned Line,
                       unsigned Column, llvm::Function *OutlinedFn,
                       llvm::Constant *OutlinedFnID, const CapturedStmt &CS)
        : ParentName(ParentName), EntryFnName(EntryFnName), DeviceID(DeviceID),
          FileID(FileID), Line(Line), Column(Column), OutlinedFn(OutlinedFn),
          OutlinedFnID(OutlinedFnID), CS(CS), ExitBB(nullptr){};
  };

  class WorkerFunctionState {
  public:
    CodeGenModule &CGM;
    const CapturedStmt &CS;

    llvm::Function *WorkerFn;
    const CGFunctionInfo *CGFI;

    WorkerFunctionState(CodeGenModule &CGM, StringRef WorkerFnName,
                        const CapturedStmt &CS)
        : CGM(CGM), CS(CS), WorkerFn(nullptr), CGFI(nullptr) {
      createWorkerFunction(WorkerFnName);
    };

  private:
    void createWorkerFunction(StringRef WorkerFnName) {
      auto &Ctx = CGM.getContext();

      // Create an worker function with no arguments.
      FunctionType::ExtInfo EI;
      CGFI = &CGM.getTypes().arrangeFreeFunctionDeclaration(
          Ctx.VoidTy, {}, EI, /*isVariadic=*/false);

      WorkerFn = llvm::Function::Create(CGM.getTypes().GetFunctionType(*CGFI),
                                        llvm::GlobalValue::InternalLinkage,
                                        WorkerFnName, &CGM.getModule());
      CGM.SetInternalFunctionAttributes(/*D=*/nullptr, WorkerFn, *CGFI);
      WorkerFn->setLinkage(llvm::GlobalValue::InternalLinkage);
      WorkerFn->addFnAttr(llvm::Attribute::NoInline);
    }
  };

  /// \brief Start a new target region.
  void enterTarget();

  /// \brief Close the current target region.
  void exitTarget();

  /// \brief Emit the worker function for the current target region.
  void emitWorkerFunction(WorkerFunctionState &WST);

  /// \brief Helper for worker function. Emit body of worker loop.
  void emitWorkerLoop(CodeGenFunction &CGF, WorkerFunctionState &WST);

  /// \brief Emit the target entry function where the master warp and all
  /// workers start.
  void emitEntryFunction(EntryFunctionState &EST, WorkerFunctionState &WST);

  /// \brief Helper for target entry function. Guide the master and worker
  /// threads to their respective locations.
  void emitEntryHeader(CodeGenFunction &CGF, EntryFunctionState &EST,
                       WorkerFunctionState &WST);

  /// \brief Signal termination of OMP execution.
  void emitEntryFooter(CodeGenFunction &CGF, EntryFunctionState &EST);

  /// \brief Gets thread id value for the current thread.
  ///
  llvm::Value *getThreadID(CodeGenFunction &CGF, SourceLocation Loc) override;

  /// \brief Emits captured variables for the outlined function for the
  /// specified OpenMP parallel directive \a D.
  void
  emitCapturedVars(CodeGenFunction &CGF, const OMPExecutableDirective &S,
                   llvm::SmallVector<llvm::Value *, 16> &CapturedVars) override;

  /// \brief Emits code for parallel or serial call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run in parallel threads. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  /// \param IfCond Condition in the associated 'if' clause, if it was
  /// specified, nullptr otherwise.
  ///
  void emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                        llvm::Value *OutlinedFn,
                        ArrayRef<llvm::Value *> CapturedVars,
                        const Expr *IfCond) override;

  /// \brief Emits outlined function for the specified OpenMP parallel directive
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  llvm::Value *
  emitParallelOutlinedFunction(const OMPExecutableDirective &D,
                               const VarDecl *ThreadIDVar,
                               OpenMPDirectiveKind InnermostKind,
                               const RegionCodeGenTy &CodeGen) override;
};

} // CodeGen namespace.
} // clang namespace.

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
