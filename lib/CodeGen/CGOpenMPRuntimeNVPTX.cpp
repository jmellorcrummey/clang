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

void CGOpenMPRuntimeNVPTX::createOffloadEntry(llvm::Constant *ID,
                                              llvm::Constant *Addr,
                                              uint64_t Size) {
  auto *F = dyn_cast<llvm::Function>(Addr);
  if (!F)
    return;
  llvm::Module *M = F->getParent();
  llvm::LLVMContext &Ctx = M->getContext();

  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode *MD = M->getOrInsertNamedMetadata("nvvm.annotations");

  llvm::Metadata *MDVals[] = {
      llvm::ConstantAsMetadata::get(F), llvm::MDString::get(Ctx, "kernel"),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 1))};
  // Append metadata to nvvm.annotations
  MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
  return;
}

/// \brief Returns specified OpenMP runtime function for the current OpenMP
/// implementation.
/// \param Function OpenMP runtime function.
/// \return Specified function.
llvm::Constant *CGOpenMPRuntimeNVPTX::createRuntimeFunction(unsigned Function) {
  llvm::Constant *RTLFn = nullptr;
  switch (static_cast<OpenMPRTLFunctionNVPTX>(Function)) {
  case OMPRTL_NVPTX__kmpc_kernel_init: {
    // Build void __kmpc_kernel_init(kmp_int32 omp_handle,
    // kmp_int32 thread_limit);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_init");
    break;
  }
  case OMPRTL_NVPTX__kmpc_serialized_parallel: {
    // Build void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_serialized_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_end_serialized_parallel: {
    // Build void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_serialized_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_prepare_parallel: {
    /// Build void __kmpc_kernel_prepare_parallel(
    /// kmp_int32 num_threads, kmp_int32 num_lanes);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_prepare_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_parallel: {
    /// Build void __kmpc_kernel_parallel(kmp_int32 num_lanes);
    llvm::Type *TypeParams[] = {CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_end_parallel: {
    /// Build void __kmpc_kernel_end_parallel();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, {}, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_end_parallel");
    break;
  }
  }

  // If we do not have a specialized version of this function, attempt to get
  // a default one.
  if (!RTLFn)
    RTLFn = CGOpenMPRuntime::createRuntimeFunction(Function);

  return RTLFn;
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

  EntryFunctionState EST;
  WorkerFunctionState WST(CGM);

  // Emit target region as a standalone region.
  auto &&CodeGen = [&EST, &WST, &CS, this](CodeGenFunction &CGF) {
    emitEntryHeader(CGF, EST, WST);
    CGF.EmitStmt(CS.getCapturedStmt());
    emitEntryFooter(CGF, EST);
  };
  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);

  // Create the worker function
  emitWorkerFunction(WST);

  // Now change the name of the worker function to correspond to this target
  // region's entry function.
  WST.WorkerFn->setName(OutlinedFn->getName() + "_worker");

  exitTarget();

  return;
}

void CGOpenMPRuntimeNVPTX::enterTarget() {
  // Nothing here for the moment.
}

void CGOpenMPRuntimeNVPTX::exitTarget() { Work.clear(); }

void CGOpenMPRuntimeNVPTX::release() {
  // Called once per module, after target processing.
  finalizeEnvironment();
}

void CGOpenMPRuntimeNVPTX::initializeEnvironment() {
  //
  // Initialize master-worker control state in shared memory.
  //

  auto DL = CGM.getDataLayout();
  ActiveWorkers = new llvm::GlobalVariable(
      CGM.getModule(), CGM.Int32Ty, /*isConstant=*/false,
      llvm::GlobalValue::CommonLinkage,
      llvm::Constant::getNullValue(CGM.Int32Ty), "__omp_num_threads", 0,
      llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);
  ActiveWorkers->setAlignment(DL.getPrefTypeAlignment(CGM.Int32Ty));

  WorkID = new llvm::GlobalVariable(
      CGM.getModule(), CGM.Int64Ty, /*isConstant=*/false,
      llvm::GlobalValue::CommonLinkage,
      llvm::Constant::getNullValue(CGM.Int64Ty), "__tgt_work_id", 0,
      llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);
  WorkID->setAlignment(DL.getPrefTypeAlignment(CGM.Int64Ty));

  // Arguments to work function.
  // This variable will be replaced once the actual size is determined,
  // after parsing all target regions.
  llvm::ArrayType *ArgsTy = llvm::ArrayType::get(CGM.IntPtrTy, 0);
  WorkArgs = new llvm::GlobalVariable(
      CGM.getModule(), ArgsTy, /*isConstant=*/false,
      llvm::GlobalValue::InternalLinkage, llvm::Constant::getNullValue(ArgsTy),
      Twine("__scratch_work_args"), 0, llvm::GlobalVariable::NotThreadLocal,
      SHARED_ADDRESS_SPACE, false);
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

void CGOpenMPRuntimeNVPTX::emitWorkerFunction(WorkerFunctionState &WST) {
  auto &Ctx = CGM.getContext();

  CodeGenFunction CGF(CGM, true);
  CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, WST.WorkerFn, *WST.CGFI, {});
  emitWorkerLoop(CGF, WST);
  CGF.FinishFunction();
}

void CGOpenMPRuntimeNVPTX::emitWorkerLoop(CodeGenFunction &CGF,
                                          WorkerFunctionState &WST) {
  //
  // The workers enter this loop and wait for parallel work from the master.
  // When the master encounters a parallel region it sets up the work + variable
  // arguments, and wakes up the workers.  The workers first check to see if
  // they are required for the parallel region, i.e., within the # of requested
  // parallel threads.  The activated workers load the variable arguments and
  // execute the parallel work.
  //

  CGBuilderTy &Bld = CGF.Builder;

  llvm::BasicBlock *AwaitBB = CGF.createBasicBlock(".await.work");
  llvm::BasicBlock *SelectWorkersBB = CGF.createBasicBlock(".select.workers");
  llvm::BasicBlock *ExecuteBB = CGF.createBasicBlock(".execute.parallel");
  llvm::BasicBlock *TerminateBB = CGF.createBasicBlock(".terminate.parallel");
  llvm::BasicBlock *BarrierBB = CGF.createBasicBlock(".barrier.parallel");
  llvm::BasicBlock *ExitBB = CGF.createBasicBlock(".sleepy.hollow");

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
  llvm::Value *ThreadID = GetNVPTXThreadID(CGF);
  llvm::Value *ActiveThread = Bld.CreateICmpSLT(
      ThreadID,
      Bld.CreateAlignedLoad(ActiveWorkers, ActiveWorkers->getAlignment()),
      "active_thread");
  Bld.CreateCondBr(ActiveThread, ExecuteBB, BarrierBB);

  // Signal start of parallel region.
  CGF.EmitBlock(ExecuteBB);
  llvm::Value *Args[] = {/*SimdNumLanes=*/Bld.getInt32(1)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_parallel),
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
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_end_parallel),
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

// Setup CUDA threads for master-worker OpenMP scheme.
void CGOpenMPRuntimeNVPTX::emitEntryHeader(CodeGenFunction &CGF,
                                           EntryFunctionState &EST,
                                           WorkerFunctionState &WST) {
  CGBuilderTy &Bld = CGF.Builder;

  // Get the master thread id.
  llvm::Value *MasterID = GetMasterThreadID(CGF);
  // Current thread's identifier.
  llvm::Value *ThreadID = GetNVPTXThreadID(CGF);

  // Setup BBs in entry function.
  llvm::BasicBlock *WorkerCheckBB = CGF.createBasicBlock(".check.for.worker");
  llvm::BasicBlock *WorkerBB = CGF.createBasicBlock(".worker");
  llvm::BasicBlock *MasterBB = CGF.createBasicBlock(".master");
  EST.ExitBB = CGF.createBasicBlock(".sleepy.hollow");

  // The head (master thread) marches on while its body of companion threads in
  // the warp go to sleep.
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
  llvm::Value *Args[] = {Bld.getInt32(/*OmpHandle=*/0), GetNVPTXThreadID(CGF)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_init),
                      Args);
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

llvm::Value *CGOpenMPRuntimeNVPTX::getThreadID(CodeGenFunction &CGF,
                                               SourceLocation Loc) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");
  return GetGlobalThreadId(CGF);
}

void CGOpenMPRuntimeNVPTX::emitCapturedVars(
    CodeGenFunction &CGF, const OMPExecutableDirective &S,
    llvm::SmallVector<llvm::Value *, 16> &CapturedVars) {
  auto CS = cast<CapturedStmt>(S.getAssociatedStmt());
  auto Var = CGF.GenerateCapturedStmtArgument(*CS);
  CapturedVars.push_back(Var.getPointer());
}

namespace {
/// \brief Base class for handling code generation inside OpenMP regions.
class CGOpenMPRegionInfo : public CodeGenFunction::CGCapturedStmtInfo {
public:
  /// \brief Kinds of OpenMP regions used in codegen.
  enum CGOpenMPRegionKind {
    /// \brief Region with outlined function for standalone 'parallel'
    /// directive.
    ParallelOutlinedRegion,
    /// \brief Region with outlined function for standalone 'task' directive.
    TaskOutlinedRegion,
    /// \brief Region for constructs that do not require function outlining,
    /// like 'for', 'sections', 'atomic' etc. directives.
    InlinedRegion,
    /// \brief Region with outlined function for standalone 'target' directive.
    TargetRegion,
  };

  CGOpenMPRegionInfo(const CapturedStmt &CS,
                     const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind,
                     bool HasCancel)
      : CGCapturedStmtInfo(CS, CR_OpenMP), RegionKind(RegionKind),
        CodeGen(CodeGen), Kind(Kind), HasCancel(HasCancel) {}

  CGOpenMPRegionInfo(const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind,
                     bool HasCancel)
      : CGCapturedStmtInfo(CR_OpenMP), RegionKind(RegionKind), CodeGen(CodeGen),
        Kind(Kind), HasCancel(HasCancel) {}

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  virtual const VarDecl *getThreadIDVariable() const = 0;

  /// \brief Emit the captured statement body.
  void EmitBody(CodeGenFunction &CGF, const Stmt *S) override;

  /// \brief Get an LValue for the current ThreadID variable.
  /// \return LValue for thread id variable. This LValue always has type int32*.
  virtual LValue getThreadIDVariableLValue(CodeGenFunction &CGF);

  CGOpenMPRegionKind getRegionKind() const { return RegionKind; }

  OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

  bool hasCancel() const { return HasCancel; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return Info->getKind() == CR_OpenMP;
  }

protected:
  CGOpenMPRegionKind RegionKind;
  RegionCodeGenTy CodeGen;
  OpenMPDirectiveKind Kind;
  bool HasCancel;
};

/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPOutlinedRegionInfo : public CGOpenMPRegionInfo {
public:
  CGOpenMPOutlinedRegionInfo(const CapturedStmt &CS, const VarDecl *ThreadIDVar,
                             const RegionCodeGenTy &CodeGen,
                             OpenMPDirectiveKind Kind, bool HasCancel)
      : CGOpenMPRegionInfo(CS, ParallelOutlinedRegion, CodeGen, Kind,
                           HasCancel),
        ThreadIDVar(ThreadIDVar) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }
  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override { return ThreadIDVar; }

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_outlined."; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() ==
               ParallelOutlinedRegion;
  }

private:
  /// \brief A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
};
}

LValue CGOpenMPRegionInfo::getThreadIDVariableLValue(CodeGenFunction &CGF) {
  return CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(getThreadIDVariable()),
      getThreadIDVariable()->getType()->castAs<PointerType>());
}

void CGOpenMPRegionInfo::EmitBody(CodeGenFunction &CGF, const Stmt * /*S*/) {
  if (!CGF.HaveInsertPoint())
    return;
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();
  {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    CodeGen(CGF);
  }
  CGF.EHStack.popTerminate();
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
  llvm::Function *Fn = cast<llvm::Function>(OutlinedFn);
  // Force inline this outlined function at its call site.
  Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  Fn->setLinkage(llvm::GlobalValue::InternalLinkage);
  auto *RTLoc = emitUpdateLocation(CGF, Loc);
  auto &&ThenGen = [this, Fn, CapturedVars](CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;

    // Prepare # of threads in parallel region based on number requested
    // by user and request for SIMD.
    llvm::Value *Args[] = {GetNumWorkers(CGF),
                           /*SimdNumLanes=*/Bld.getInt32(1)};
    llvm::Value *ParallelThreads = CGF.EmitRuntimeCall(
        createRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_prepare_parallel), Args);

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
    auto ID = Bld.CreatePtrToInt(Fn, WorkID->getType()->getElementType());
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
  auto &&ElseGen = [this, Fn, CapturedVars, RTLoc, Loc](CodeGenFunction &CGF) {
    auto DL = CGM.getDataLayout();
    auto ThreadID = getThreadID(CGF, Loc);
    // Build calls:
    // __kmpc_serialized_parallel(&Loc, GTid);
    llvm::Value *Args[] = {RTLoc, ThreadID};
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL_NVPTX__kmpc_serialized_parallel),
                        Args);

    // OutlinedFn(&GTid, &zero, CapturedStruct);
    auto ThreadIDAddr = emitThreadIDAddress(CGF, Loc);
    Address ZeroAddr =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ ".zero.addr");
    CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
    llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
    OutlinedFnArgs.push_back(ThreadIDAddr.getPointer());
    OutlinedFnArgs.push_back(ZeroAddr.getPointer());
    OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
    CGF.EmitCallOrInvoke(Fn, OutlinedFnArgs);

    // __kmpc_end_serialized_parallel(&Loc, GTid);
    llvm::Value *EndArgs[] = {emitUpdateLocation(CGF, Loc), ThreadID};
    CGF.EmitRuntimeCall(
        createRuntimeFunction(OMPRTL_NVPTX__kmpc_end_serialized_parallel), EndArgs);
  };
  if (IfCond) {
    emitOMPIfClause(CGF, IfCond, ThenGen, ElseGen);
  } else {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    ThenGen(CGF);
  }
}

//
// Generate optimized code resembling static schedule with chunk size of 1
// whenever the standard gives us freedom.  This allows maximum coalescing on
// the NVPTX target.
//
bool CGOpenMPRuntimeNVPTX::generateCoalescedSchedule(
    OpenMPScheduleClauseKind ScheduleKind, bool ChunkSizeOne,
    bool ordered) const {
  return !ordered && (ScheduleKind == OMPC_SCHEDULE_unknown ||
                      ScheduleKind == OMPC_SCHEDULE_auto ||
                      (ScheduleKind == OMPC_SCHEDULE_static && ChunkSizeOne));
}

bool CGOpenMPRuntimeNVPTX::requiresBarrier(const OMPLoopDirective &S) const {
  const bool Ordered = S.getSingleClause<OMPOrderedClause>() != nullptr;
  OpenMPScheduleClauseKind ScheduleKind = OMPC_SCHEDULE_unknown;
  if (auto *C = S.getSingleClause<OMPScheduleClause>())
    ScheduleKind = C->getScheduleKind();
  return Ordered || ScheduleKind == OMPC_SCHEDULE_dynamic ||
         ScheduleKind == OMPC_SCHEDULE_guided;
}
