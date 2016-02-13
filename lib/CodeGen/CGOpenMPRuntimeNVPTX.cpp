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

#include "CGOpenMPRuntimeNVPTX.h"

using namespace clang;
using namespace CodeGen;

CGOpenMPRuntimeNVPTX::CGOpenMPRuntimeNVPTX(CodeGenModule &CGM)
    : CGOpenMPRuntime(CGM), ActiveWorkers(nullptr), WorkID(nullptr),
      WorkArgs(nullptr), MaxWorkArgs(0) {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP NVPTX can only handle device code.");

  // Called once per module during initialization.
  initializeEnvironment();
}

void CGOpenMPRuntimeNVPTX::release() {
  // Called once per module, after target processing.
  finalizeEnvironment();
}

void CGOpenMPRuntimeNVPTX::initializeEnvironment() {
  //
  // Initialize master-worker control state in shared memory.
  //

  auto DL = CGM.getDataLayout();
  // Number of requested OMP threads in parallel region.
  ActiveWorkers = new llvm::GlobalVariable(
      CGM.getModule(), CGM.Int32Ty, false, llvm::GlobalValue::CommonLinkage,
      llvm::Constant::getNullValue(CGM.Int32Ty), "__omp_num_threads", 0,
      llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);
  ActiveWorkers->setAlignment(DL.getPrefTypeAlignment(CGM.Int32Ty));

  // Work function ID.
  WorkID = new llvm::GlobalVariable(
      CGM.getModule(), CGM.Int64Ty, false, llvm::GlobalValue::CommonLinkage,
      llvm::Constant::getNullValue(CGM.Int64Ty), "__tgt_work_id", 0,
      llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);
  WorkID->setAlignment(DL.getPrefTypeAlignment(CGM.Int64Ty));

  // Arguments to work function.
  // This variable will be replaced once the actual size is determined,
  // after parsing all target regions.
  llvm::ArrayType *ArgsTy = llvm::ArrayType::get(CGM.IntPtrTy, 1);
  WorkArgs = new llvm::GlobalVariable(
      CGM.getModule(), ArgsTy, false, llvm::GlobalValue::InternalLinkage,
      llvm::Constant::getNullValue(ArgsTy), Twine("__scratch_work_args"), 0,
      llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);
  WorkArgs->setAlignment(DL.getPrefTypeAlignment(CGM.IntPtrTy));
}

void CGOpenMPRuntimeNVPTX::finalizeEnvironment() {
  // Update the size of the work_args structure based on the maximum number
  // of captured variables seen in this compilation unit so far.
  auto DL = CGM.getDataLayout();
  llvm::ArrayType *ArgsTy = llvm::ArrayType::get(CGM.IntPtrTy, MaxWorkArgs);
  llvm::GlobalVariable *FinalWorkArgs = new llvm::GlobalVariable(
      CGM.getModule(), ArgsTy, false, llvm::GlobalValue::CommonLinkage,
      llvm::Constant::getNullValue(ArgsTy), Twine("__tgt_work_args"), 0,
      llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);
  FinalWorkArgs->setAlignment(DL.getPrefTypeAlignment(CGM.IntPtrTy));

  // Replace uses manually since the type has changed (size of array).
  // The original reference will be optimized out.
  while (!WorkArgs->use_empty()) {
    auto &U = *WorkArgs->use_begin();
    // Must handle Constants specially, we cannot call replaceUsesOfWith on a
    // constant because they are uniqued.
    if (llvm::Constant *C = dyn_cast<llvm::Constant>(U.getUser())) {
      if (!isa<llvm::GlobalValue>(C)) {
        C->handleOperandChange(WorkArgs, FinalWorkArgs);
        continue;
      }
    }

    U.set(FinalWorkArgs);
  }
}

void CGOpenMPRuntimeNVPTX::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry) {
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

void CGOpenMPRuntimeNVPTX::enterTarget() {
  // Nothing here for the moment.
}

void CGOpenMPRuntimeNVPTX::exitTarget() { Work.clear(); }

void CGOpenMPRuntimeNVPTX::emitWorkerFunction(WorkerFunctionState &WST) {
  auto &Ctx = CGM.getContext();

  CodeGenFunction CGF(CGM, true);
  CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, WST.WorkerFn, *WST.CGFI, {});
  emitWorkerLoop(CGF, WST);
  CGF.FinishFunction();
}

void CGOpenMPRuntimeNVPTX::emitWorkerLoop(CodeGenFunction &CGF,
                                          WorkerFunctionState &WST) {
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
  llvm::Value *ThreadID = Bld.CreateCall(GetNVPTXThreadID(), {}, "tid");
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
    llvm::Value *WorkIDMatch =
        Bld.CreateICmpEQ(Bld.CreateAlignedLoad(WorkID, WorkID->getAlignment()),
                         ID, "work_match");

    llvm::BasicBlock *ExecuteFNBB = CGF.createBasicBlock(".execute.fn");
    llvm::BasicBlock *CheckNextBB = CGF.createBasicBlock(".check.next");
    Bld.CreateCondBr(WorkIDMatch, ExecuteFNBB, CheckNextBB);

    // Execute this outlined function.
    CGF.EmitBlock(ExecuteFNBB);
    llvm::SmallVector<llvm::Value *, 20> FnArgs;
    // First two arguments are not used on the device.
    Address ZeroAddr =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ ".zero.addr");
    CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
    FnArgs.push_back(ZeroAddr.getPointer());
    FnArgs.push_back(llvm::Constant::getNullValue(CGM.Int32Ty->getPointerTo()));

    // Cast the shared memory structure work_args and pass it as the third
    // argument, i.e., the capture structure.
    auto Fn = cast<llvm::Function>(W);
    auto CaptureType = std::next(std::next(Fn->arg_begin()))->getType();
    auto Capture = Bld.CreateAddrSpaceCast(WorkArgs, CaptureType);
    FnArgs.push_back(Capture);

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

void CGOpenMPRuntimeNVPTX::emitEntryFunction(EntryFunctionState &EST,
                                             WorkerFunctionState &WST) {
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
void CGOpenMPRuntimeNVPTX::emitEntryHeader(CodeGenFunction &CGF,
                                           EntryFunctionState &EST,
                                           WorkerFunctionState &WST) {
  CGBuilderTy &Bld = CGF.Builder;

  // Get the master thread id.
  llvm::Value *MasterID = GetMasterThreadID(CGF);
  // Current thread's identifier.
  llvm::Value *ThreadID = Bld.CreateCall(GetNVPTXThreadID(), {}, "tid");

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
      Bld.CreateCall(GetNVPTXThreadID(), {}, "thread_limit")};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_kernel_init), Args);
}

void CGOpenMPRuntimeNVPTX::emitEntryFooter(CodeGenFunction &CGF,
                                           EntryFunctionState &EST) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::BasicBlock *TerminateBB = CGF.createBasicBlock(".termination.notifier");
  CGF.EmitBranch(TerminateBB);

  CGF.EmitBlock(TerminateBB);
  // Signal termination condition.
  Bld.CreateAlignedStore(
      llvm::Constant::getNullValue(WorkID->getType()->getElementType()), WorkID,
      WorkID->getAlignment());
  // Barrier to terminate worker threads.
  SyncCTAThreads(CGF);
  // Master thread jumps to exit point.
  CGF.EmitBranch(EST.ExitBB);

  CGF.EmitBlock(EST.ExitBB);
}

void CGOpenMPRuntimeNVPTX::emitCapturedVars(
    CodeGenFunction &CGF, const OMPExecutableDirective &S,
    llvm::SmallVector<llvm::Value *, 16> &CapturedVars) {
  auto CS = cast<CapturedStmt>(S.getAssociatedStmt());
  auto Var = CGF.GenerateCapturedStmtArgument(*CS);
  CapturedVars.push_back(Var.getPointer());
}

llvm::Value *CGOpenMPRuntimeNVPTX::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  assert(ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 *");
  const CapturedStmt *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  CodeGenFunction CGF(CGM, true);
  bool HasCancel = false;
  if (auto *OPD = dyn_cast<OMPParallelDirective>(&D))
    HasCancel = OPD->hasCancel();
  else if (auto *OPSD = dyn_cast<OMPParallelSectionsDirective>(&D))
    HasCancel = OPSD->hasCancel();
  else if (auto *OPFD = dyn_cast<OMPParallelForDirective>(&D))
    HasCancel = OPFD->hasCancel();
  CGOpenMPOutlinedRegionInfo CGInfo(*CS, ThreadIDVar, CodeGen, InnermostKind,
                                    HasCancel);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
  // The outlined function takes as arguments the global_tid, bound_tid,
  // and a capture structure created from the captured variables.
  return CGF.GenerateCapturedStmtFunction(*CS);
}

void CGOpenMPRuntimeNVPTX::emitParallelCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> CapturedVars, const Expr *IfCond) {
  if (!CGF.HaveInsertPoint())
    return;
  auto &&ThenGen = [this, OutlinedFn, CapturedVars](CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    auto DL = CGM.getDataLayout();

    llvm::Function *Fn = cast<llvm::Function>(OutlinedFn);
    // Force inline this outlined function at its call site.
    Fn->addFnAttr(llvm::Attribute::AlwaysInline);
    Fn->setLinkage(llvm::GlobalValue::InternalLinkage);

    // Prepare # of threads in parallel region based on number requested
    // by user and request for SIMD.
    llvm::Value *Args[] = {GetNumWorkers(CGF),
                           /*SimdNumLanes=*/Bld.getInt32(1)};
    llvm::Value *ParallelThreads = CGF.EmitRuntimeCall(
        createRuntimeFunction(OMPRTL__kmpc_kernel_prepare_parallel), Args);

    // Set number of workers to activate.
    Bld.CreateAlignedStore(ParallelThreads, ActiveWorkers,
                           ActiveWorkers->getAlignment());

    // Copy variables in the capture buffer to the shared workargs structure for
    // use by the workers.  The extraneous capture buffer will be optimized out
    // by llvm.
    auto Capture = CapturedVars.front();
    auto CaptureType = Capture->getType()->getArrayElementType();
    auto WorkArgsPtr = Bld.CreateAddrSpaceCast(WorkArgs, Capture->getType());
    for (unsigned idx = 0; idx < CaptureType->getStructNumElements(); idx++) {
      auto Src = Bld.CreateConstInBoundsGEP2_32(CaptureType, Capture, 0, idx);
      auto Dst =
          Bld.CreateConstInBoundsGEP2_32(CaptureType, WorkArgsPtr, 0, idx);
      Bld.CreateDefaultAlignedStore(Bld.CreateDefaultAlignedLoad(Src), Dst);
    }

    // Indicate parallel function to execute.
    auto ID =
        Bld.CreatePtrToInt(OutlinedFn, WorkID->getType()->getElementType());
    Bld.CreateAlignedStore(ID, WorkID, WorkID->getAlignment());

    // Activate workers.
    SyncCTAThreads(CGF);

    // Barrier at end of parallel region.
    SyncCTAThreads(CGF);

    // Remember for post-processing in worker loop.
    Work.push_back(Fn);
    MaxWorkArgs =
        std::max(MaxWorkArgs, (size_t)CaptureType->getStructNumElements());
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
