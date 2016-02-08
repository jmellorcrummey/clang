//===---- CGOpenMPRuntimeNVPTX.cpp - Interface to OpenMP NVPTX Runtimes ---===//
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

//#include "CGOpenMPRuntimeNVPTX.h"
#include "CGOpenMPRuntime.h"
#include "CGOpenMPRuntimeCommon.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/IR/CallSite.h"

using namespace clang;
using namespace CodeGen;

class CGOpenMPRuntimeNVPTX : public CGOpenMPRuntime {
private:
  llvm::Function *GetCudaWarpSize() {
    return llvm::Intrinsic::getDeclaration(
        &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize);
  }

  llvm::Function *GetCudaThreadID() {
    return llvm::Intrinsic::getDeclaration(
        &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
  }

  llvm::Function *GetCudaNumThreads() {
    return llvm::Intrinsic::getDeclaration(
        &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  }

  llvm::Function *GetCudaCTABarrier() {
    return llvm::Intrinsic::getDeclaration(&CGM.getModule(),
                                           llvm::Intrinsic::nvvm_barrier_n);
  }

  llvm::Function *GetCudaBarrier() {
    return llvm::Intrinsic::getDeclaration(&CGM.getModule(),
                                           llvm::Intrinsic::nvvm_barrier);
  }

public:
  explicit CGOpenMPRuntimeNVPTX(CodeGenModule &CGM)
      : CGOpenMPRuntime(CGM), ActiveWorkers(nullptr), ThreadLimit(nullptr),
        WorkID(nullptr), WorkArgs(nullptr) {
    if (!CGM.getLangOpts().OpenMPIsDevice)
      llvm_unreachable("OpenMP NVPTX can only handle device code.");
  }

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
                                          bool IsOffloadEntry) override {
    if (!IsOffloadEntry) // Nothing to do.
      return;

    assert(!ParentName.empty() && "Invalid target region parent name!");

    const CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());

    enterTarget();

    // Name the target region.
    SmallString<256> EntryFnName;
    unsigned DeviceID;
    unsigned FileID;
    unsigned Line;
    unsigned Column;
    getUniqueTargetEntryName(D, ParentName, DeviceID, FileID, Line, Column,
                             EntryFnName);

    // Create the entry function and populate the worker loop
    EntryFunctionState EST(ParentName, EntryFnName, DeviceID, FileID, Line,
                           Column, OutlinedFn, OutlinedFnID, CS);
    SmallString<256> WorkerFnName = EntryFnName;
    WorkerFnName += "_worker";
    WorkerFunctionState WST(CGM, WorkerFnName, CS);

    // Create the entry point where master invokes workers
    emitEntryFunction(EST, WST);

    // Create the worker function
    emitWorkerFunction(WST);

    OutlinedFn = EST.OutlinedFn;
    OutlinedFnID = EST.OutlinedFnID;

    exitTarget();

    return;
  }

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
  // TODO: fix!  Multiple kernels, and even the same kernel may run
  // concurrently.
  llvm::GlobalVariable *ThreadLimit;
  llvm::GlobalVariable *WorkID;
  llvm::GlobalVariable *WorkArgs;

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
      // TODO: Pass entry arguments to worker and reuse instead of passing via
      // shared memory.
      FunctionType::ExtInfo EI;
      CGFI = &CGM.getTypes().arrangeFreeFunctionDeclaration(
          Ctx.VoidTy, {}, EI, /*isVariadic=*/false);

      WorkerFn = llvm::Function::Create(CGM.getTypes().GetFunctionType(*CGFI),
                                        llvm::GlobalValue::InternalLinkage,
                                        WorkerFnName, &CGM.getModule());
      CGM.SetInternalFunctionAttributes(/*D=*/nullptr, WorkerFn, *CGFI);
      WorkerFn->setLinkage(llvm::GlobalValue::PrivateLinkage);
      WorkerFn->addFnAttr(llvm::Attribute::NoInline);
    }
  };

  void enterTarget() {
    auto DL = CGM.getDataLayout();

    // Initialize master-worker control state in shared memory.  It should be
    // shared across all target regions.

    // Number of requested OMP threads in parallel region.
    if (!ActiveWorkers) {
      ActiveWorkers = new llvm::GlobalVariable(
          CGM.getModule(), CGM.Int32Ty, false, llvm::GlobalValue::CommonLinkage,
          llvm::Constant::getNullValue(CGM.Int32Ty), "__omp_num_threads", 0,
          llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);
      ActiveWorkers->setAlignment(DL.getPrefTypeAlignment(CGM.Int32Ty));
    }

    // Add global for thread_limit that is kept updated by the CUDA offloading
    // RTL.  Value of 0 causes runtime to use default.
    ThreadLimit = new llvm::GlobalVariable(
        CGM.getModule(), CGM.Int32Ty, false, llvm::GlobalValue::ExternalLinkage,
        llvm::Constant::getNullValue(CGM.Int32Ty), Twine("__omp_thread_limit"));
    ThreadLimit->setAlignment(DL.getPrefTypeAlignment(CGM.Int32Ty));

    // Work function ID.
    if (!WorkID) {
      WorkID = new llvm::GlobalVariable(
          CGM.getModule(), CGM.Int64Ty, false, llvm::GlobalValue::CommonLinkage,
          llvm::Constant::getNullValue(CGM.Int64Ty), "__tgt_work_id", 0,
          llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);
      WorkID->setAlignment(DL.getPrefTypeAlignment(CGM.Int64Ty));
    }

    // Arguments to work function.
    if (!WorkArgs) {
      llvm::ArrayType *ArgsTy = llvm::ArrayType::get(CGM.IntPtrTy, 20);
      WorkArgs = new llvm::GlobalVariable(
          CGM.getModule(), ArgsTy, false, llvm::GlobalValue::CommonLinkage,
          llvm::Constant::getNullValue(ArgsTy), Twine("__tgt_work_args"), 0,
          llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);
      WorkArgs->setAlignment(DL.getPrefTypeAlignment(CGM.IntPtrTy));
    }
  }

  void exitTarget() { Work.clear(); }

  void emitWorkerFunction(WorkerFunctionState &WST) {
    auto &Ctx = CGM.getContext();

    CodeGenFunction CGF(CGM, true);
    CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, WST.WorkerFn, *WST.CGFI, {});
    emitWorkerLoop(CGF, WST);
    CGF.FinishFunction();
  }

  void emitWorkerLoop(CodeGenFunction &CGF, WorkerFunctionState &WST) {
    CGBuilderTy &Bld = CGF.Builder;

    llvm::BasicBlock *AwaitBB = CGF.createBasicBlock(".await.work");
    llvm::BasicBlock *SelectWorkersBB = CGF.createBasicBlock(".select.workers");
    llvm::BasicBlock *ExecuteBB = CGF.createBasicBlock(".execute.parallel");
    llvm::BasicBlock *TerminateBB = CGF.createBasicBlock(".terminate.parallel");
    llvm::BasicBlock *BarrierBB = CGF.createBasicBlock(".barrier.parallel");
    llvm::BasicBlock *ExitBB = CGF.createBasicBlock(".pearly.gates");

    CGF.EmitBranch(AwaitBB);

    // Workers wait for work from master.
    CGF.EmitBlock(AwaitBB);
    // Wait for parallel work
    SyncCTAThreads(CGF);
    // On termination condition (workid == 0), exit loop.
    llvm::Value *ShouldTerminate = Bld.CreateICmpEQ(
        Bld.CreateAlignedLoad(WorkID, WorkID->getAlignment()),
        llvm::Constant::getNullValue(WorkID->getType()->getElementType()),
        "should_terminate");
    Bld.CreateCondBr(ShouldTerminate, ExitBB, SelectWorkersBB);

    // Activate requested workers.
    CGF.EmitBlock(SelectWorkersBB);
    llvm::Value *ThreadID = Bld.CreateCall(GetCudaThreadID(), {}, "tid");
    llvm::Value *ActiveThread = Bld.CreateICmpSLT(
        ThreadID,
        Bld.CreateAlignedLoad(ActiveWorkers, ActiveWorkers->getAlignment()),
        "active_thread");
    Bld.CreateCondBr(ActiveThread, ExecuteBB, BarrierBB);

    // Signal start of parallel region.
    CGF.EmitBlock(ExecuteBB);
    llvm::Value *Args[] = {/*SimdNumLanes=*/Bld.getInt32(1)};
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_kernel_parallel),
                        Args);

    // Process work items: outlined parallel functions.
    for (auto W : Work) {
      // Try to match this outlined function.
      auto ID = Bld.CreatePtrToInt(W, WorkID->getType()->getElementType());
      llvm::Value *WorkIDMatch = Bld.CreateICmpEQ(
          Bld.CreateAlignedLoad(WorkID, WorkID->getAlignment()), ID,
          "work_match");

      llvm::BasicBlock *ExecuteFNBB = CGF.createBasicBlock(".execute.fn");
      llvm::BasicBlock *CheckNextBB = CGF.createBasicBlock(".check.next");
      Bld.CreateCondBr(WorkIDMatch, ExecuteFNBB, CheckNextBB);

      // Execute this outlined function.
      CGF.EmitBlock(ExecuteFNBB);
      llvm::SmallVector<llvm::Value *, 20> FnArgs;
      // First two arguments are not used and don't need to be retrieved.
      Address ZeroAddr =
          CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                               /*Name*/ ".zero.addr");
      CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
      FnArgs.push_back(ZeroAddr.getPointer());
      FnArgs.push_back(
          llvm::Constant::getNullValue(CGM.Int32Ty->getPointerTo()));
      auto Fn = cast<llvm::Function>(W);
      auto begin = std::next(std::next(Fn->arg_begin()));
      unsigned idx = 0;
      // Load outlined function arguments from memory.
      for (llvm::Function::const_arg_iterator ai = begin, ae = Fn->arg_end();
           ai != ae; ++ai) {
        auto StoredArg = Bld.CreateConstInBoundsGEP2_32(
            WorkArgs->getValueType(), WorkArgs, 0, idx++);
        llvm::Value *StoredVal =
            Bld.CreateAlignedLoad(StoredArg, WorkArgs->getAlignment());
        if (ai->getType() != StoredVal->getType()) {
          StoredVal = Bld.CreateBitOrPointerCast(StoredVal, ai->getType());
        }
        FnArgs.push_back(StoredVal);
      }
      // Insert call to work function.
      CGF.EmitCallOrInvoke(Fn, FnArgs);
      // Go to end of parallel region.
      CGF.EmitBranch(TerminateBB);

      CGF.EmitBlock(CheckNextBB);
    }

    // Signal end of parallel region.
    CGF.EmitBlock(TerminateBB);
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_kernel_end_parallel),
                        ArrayRef<llvm::Value *>());
    CGF.EmitBranch(BarrierBB);

    // All active and inactive workers wait at a barrier after parallel region.
    CGF.EmitBlock(BarrierBB);
    // Barrier after parallel region.
    SyncCTAThreads(CGF);
    CGF.EmitBranch(AwaitBB);

    // Exit target region.
    CGF.EmitBlock(ExitBB);
  }

  void emitEntryFunction(EntryFunctionState &EST, WorkerFunctionState &WST) {
    // Emit target region as a standalone region.
    auto &&CodeGen = [&EST, &WST, this](CodeGenFunction &CGF) {
      emitEntryHeader(CGF, EST, WST);
      CGF.EmitStmt(EST.CS.getCapturedStmt());
      emitEntryFooter(CGF, EST);
    };

    CodeGenFunction CGF(CGM, true);
    CGOpenMPTargetRegionInfo CGInfo(EST.CS, CodeGen, EST.EntryFnName);
    CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
    EST.OutlinedFn = CGF.GenerateOpenMPCapturedStmtFunction(EST.CS);

    // When emitting code for the device, the ID has to be the function address
    // so that it can retrieved from the offloading entry and launched by the
    // runtime library. We also mark the outlined function to have external
    // linkage in case we are emitting code for the device, because these
    // functions will be entry points to the device.
    EST.OutlinedFnID =
        llvm::ConstantExpr::getBitCast(EST.OutlinedFn, CGM.Int8PtrTy);
    EST.OutlinedFn->setLinkage(llvm::GlobalValue::ExternalLinkage);

    // Register the information for the entry associated with this target
    // region.
    OffloadEntriesInfoManager.registerTargetRegionEntryInfo(
        EST.DeviceID, EST.FileID, EST.ParentName, EST.Line, EST.Column,
        EST.OutlinedFn, EST.OutlinedFnID);
  }

  // Setup CUDA threads for master-worker OpenMP scheme.
  void emitEntryHeader(CodeGenFunction &CGF, EntryFunctionState &EST,
                       WorkerFunctionState &WST) {
    CGBuilderTy &Bld = CGF.Builder;

    // Get the master thread id.
    llvm::Value *MasterID = GetMasterThreadID(CGF);
    // Current thread's identifier.
    llvm::Value *ThreadID = Bld.CreateCall(GetCudaThreadID(), {}, "tid");

    // Setup BBs in entry function.
    llvm::BasicBlock *WorkerCheckBB = CGF.createBasicBlock(".check.for.worker");
    llvm::BasicBlock *WorkerBB = CGF.createBasicBlock(".worker");
    llvm::BasicBlock *MasterBB = CGF.createBasicBlock(".master");
    EST.ExitBB = CGF.createBasicBlock(".pearly.gates");

    // All threads in master warp except for the master thread are sent to
    // purgatory.
    llvm::Value *ShouldDie =
        Bld.CreateICmpUGT(ThreadID, MasterID, "excess_in_master_warp");
    Bld.CreateCondBr(ShouldDie, EST.ExitBB, WorkerCheckBB);

    // Select worker threads...
    CGF.EmitBlock(WorkerCheckBB);
    llvm::Value *IsWorker = Bld.CreateICmpULT(ThreadID, MasterID, "is_worker");
    Bld.CreateCondBr(IsWorker, WorkerBB, MasterBB);

    // ... and send to worker loop, awaiting parallel invocation.
    CGF.EmitBlock(WorkerBB);
    llvm::SmallVector<llvm::Value *, 16> WorkerVars;
    for (auto &I : CGF.CurFn->args()) {
      WorkerVars.push_back(&I);
    }

    CGF.EmitCallOrInvoke(WST.WorkerFn, {});
    CGF.EmitBranch(EST.ExitBB);

    // Only master thread executes subsequent serial code.
    CGF.EmitBlock(MasterBB);

    // First action in sequential region:
    // Initialize the state of the OpenMP runtime library on the GPU.
    llvm::Value *Args[] = {
        Bld.getInt32(/*OmpHandle=*/0),
        Bld.CreateAlignedLoad(ThreadLimit, ThreadLimit->getAlignment())};
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_kernel_init), Args);
  }

  void emitEntryFooter(CodeGenFunction &CGF, EntryFunctionState &EST) {
    CGBuilderTy &Bld = CGF.Builder;
    llvm::BasicBlock *TerminateBB =
        CGF.createBasicBlock(".termination.notifier");
    CGF.EmitBranch(TerminateBB);

    CGF.EmitBlock(TerminateBB);
    // Signal termination condition.
    Bld.CreateAlignedStore(
        llvm::Constant::getNullValue(WorkID->getType()->getElementType()),
        WorkID, WorkID->getAlignment());
    // Barrier to terminate worker threads.
    SyncCTAThreads(CGF);
    // Master thread jumps to exit point.
    CGF.EmitBranch(EST.ExitBB);

    CGF.EmitBlock(EST.ExitBB);
  }

  void emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                        llvm::Value *OutlinedFn,
                        ArrayRef<llvm::Value *> CapturedVars,
                        const Expr *IfCond) override {
    if (!CGF.HaveInsertPoint())
      return;
    auto &&ThenGen = [this, OutlinedFn, CapturedVars](CodeGenFunction &CGF) {
      CGBuilderTy &Bld = CGF.Builder;
      auto &Ctx = CGF.getContext();

      // Prepare # of threads in parallel region based on number requested
      // by user and request for SIMD.
      llvm::Value *Args[] = {GetNumWorkers(CGF),
                             /*SimdNumLanes=*/Bld.getInt32(1)};
      llvm::Value *ParallelThreads = CGF.EmitRuntimeCall(
          createRuntimeFunction(OMPRTL__kmpc_kernel_prepare_parallel), Args);

      // Set number of workers to activate.
      Bld.CreateAlignedStore(ParallelThreads, ActiveWorkers,
                             ActiveWorkers->getAlignment());

      assert(CapturedVars.size() < 20 &&
             "FIXME: Parallel region has more than 20 captured vars.");

      int idx = 0;
      for (auto Var : CapturedVars) {
        auto Arg = CGF.Builder.CreateConstInBoundsGEP2_32(
            WorkArgs->getValueType(), WorkArgs, 0, idx++);
        Address ArgAddr(Arg, Ctx.getTypeAlignInChars(Ctx.VoidPtrTy));
        if (Var->getType() != Arg->getType()) {
          ArgAddr = Bld.CreateElementBitCast(ArgAddr, Var->getType());
        }
        Bld.CreateStore(Var, ArgAddr);
      }

      auto ID =
          Bld.CreatePtrToInt(OutlinedFn, WorkID->getType()->getElementType());
      Bld.CreateAlignedStore(ID, WorkID, WorkID->getAlignment());

      // Activate workers.
      SyncCTAThreads(CGF);

      // Barrier at end of parallel region.
      SyncCTAThreads(CGF);

      llvm::Function *Fn = cast<llvm::Function>(OutlinedFn);
      // Force inline this outlined function at its call site.
      Fn->addFnAttr(llvm::Attribute::AlwaysInline);

      Work.push_back(Fn);
    };
    //    auto &&ElseGen = [this, OutlinedFn, CapturedVars, RTLoc,
    //                      Loc](CodeGenFunction &CGF) {
    //      auto ThreadID = getThreadID(CGF, Loc);
    //      // Build calls:
    //      // __kmpc_serialized_parallel(&Loc, GTid);
    //      llvm::Value *Args[] = {RTLoc, ThreadID};
    //      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_serialized_parallel),
    //                          Args);
    //
    //      // OutlinedFn(&GTid, &zero, CapturedStruct);
    //      auto ThreadIDAddr = emitThreadIDAddress(CGF, Loc);
    //      Address ZeroAddr =
    //        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
    //                             /*Name*/ ".zero.addr");
    //      CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
    //      llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
    //      OutlinedFnArgs.push_back(ThreadIDAddr.getPointer());
    //      OutlinedFnArgs.push_back(ZeroAddr.getPointer());
    //      OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
    //      CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);
    //
    //      // __kmpc_end_serialized_parallel(&Loc, GTid);
    //      llvm::Value *EndArgs[] = {emitUpdateLocation(CGF, Loc), ThreadID};
    //      CGF.EmitRuntimeCall(
    //          createRuntimeFunction(OMPRTL__kmpc_end_serialized_parallel),
    //          EndArgs);
    //    };
    //    if (IfCond) {
    //      emitOMPIfClause(CGF, IfCond, ThenGen, ElseGen);
    //    } else {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    ThenGen(CGF);
    //    }
  }

  // The master thread id is the first thread (lane) of the last warp.
  // Thread id is 0 indexed.
  // E.g: If NumThreads is 33, master id is 32.
  //      If NumThreads is 64, master id is 32.
  //      If NumThreads is 1024, master id is 992.
  llvm::Value *GetMasterThreadID(CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    llvm::Value *NumThreads = Bld.CreateCall(GetCudaNumThreads(), {});

    // We assume that the warp size is a multiple of 2.
    llvm::Value *Mask =
        Bld.CreateSub(Bld.CreateCall(GetCudaWarpSize(), {}), Bld.getInt32(1));

    return Bld.CreateAnd(Bld.CreateSub(NumThreads, Bld.getInt32(1)),
                         Bld.CreateNot(Mask), "master_tid");
  }

  // Get number of workers after subtracting the master warp
  llvm::Value *GetNumWorkers(CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    return Bld.CreateSub(GetMasterThreadID(CGF), Bld.getInt32(1),
                         "num_workers");
  }

  // Sync all threads in CTA
  void SyncCTAThreads(CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    llvm::Value *Args[] = {Bld.getInt32(CTA_BARRIER)};
    Bld.CreateCall(GetCudaCTABarrier(), Args);
  }
};

namespace CGOpenMPCommon{
  CGOpenMPRuntime *createCGOpenMPRuntimeNVPTX(CodeGenModule &CGM){
    return new CGOpenMPRuntimeNVPTX(CGM);
  }
} // CGOpenMPCommon namespace.
