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
#include "CGCleanup.h"
#include "clang/AST/DeclOpenMP.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtOpenMP.h"

using namespace clang;
using namespace CodeGen;

namespace {
enum OpenMPRTLFunctionNVPTX {
  /// \brief Call to void __kmpc_kernel_init(kmp_int32 omp_handle,
  /// kmp_int32 thread_limit);
  OMPRTL_NVPTX__kmpc_kernel_init,
  /// \brief Call to void __kmpc_kernel_deinit();
  OMPRTL_NVPTX__kmpc_kernel_deinit,
  // Call to void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
  // global_tid);
  OMPRTL_NVPTX__kmpc_serialized_parallel,
  // Call to void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
  // global_tid);
  OMPRTL_NVPTX__kmpc_end_serialized_parallel,
  /// \brief Call to void __kmpc_kernel_prepare_parallel(
  /// void *outlined_function, void **args, kmp_int32 nArgs);
  OMPRTL_NVPTX__kmpc_kernel_prepare_parallel,
  /// \brief Call to bool __kmpc_kernel_parallel(
  /// void **outlined_function, void **args);
  OMPRTL_NVPTX__kmpc_kernel_parallel,
  /// \brief Call to void __kmpc_kernel_end_parallel();
  OMPRTL_NVPTX__kmpc_kernel_end_parallel,
  /// \brief Call to bool __kmpc_kernel_convergent_parallel(
  /// void *buffer, bool *IsFinal, kmpc_int32 *LaneSource);
  OMPRTL_NVPTX__kmpc_kernel_convergent_parallel,
  /// \brief Call to void __kmpc_kernel_end_convergent_parallel(
  /// void *buffer);
  OMPRTL_NVPTX__kmpc_kernel_end_convergent_parallel,
  /// \brief Call to int32_t __kmpc_warp_active_thread_mask();
  OMPRTL_NVPTX__kmpc_warp_active_thread_mask,
//  /// \brief Call to void * malloc(size_t size);
//  OMPRTL_NVPTX__malloc,
//  /// \brief Call to void free(void *ptr);
//  OMPRTL_NVPTX__free,
  /// \brief Call to void __kmpc_initialize_data_sharing_environment(__kmpc_data_sharing_slot *RootS, __kmpc_data_sharing_slot **SharedS, void **SharedD, size_t InitialDataSize);
  OMPRTL_NVPTX__kmpc_initialize_data_sharing_environment,
  /// \brief Call to void* __kmpc_data_sharing_environment_begin(__kmpc_data_sharing_slot **SharedS, void **SharedD, __kmpc_data_sharing_slot **SavedSharedS, void **SavedSharedD, size_t SharingDataSize, size_t SharingDefaultDataSize, int32_t *ReuseData);
  OMPRTL_NVPTX__kmpc_data_sharing_environment_begin,
  /// \brief Call to void __kmpc_data_sharing_environment_end( __kmpc_data_sharing_slot **SharedS, void **SharedD, __kmpc_data_sharing_slot **SavedSharedS, void **SavedSharedD);
  OMPRTL_NVPTX__kmpc_data_sharing_environment_end,
};

// NVPTX Address space
enum ADDRESS_SPACE {
  ADDRESS_SPACE_SHARED = 3,
};

enum STATE_SIZE {
  TASK_STATE_SIZE = 48,
};

enum DATA_SHARING_SIZES {
  // The maximum number of workers in a kernel.
  DS_Max_Worker_Threads = 992,
  // The size reserved for data in a shared memory slot.
  DS_Slot_Size = 4,
  // The maximum number of threads in a worker warp.
  DS_Max_Worker_Warp_Size = 32,
  // The number of bits required to represent the maximum number of threads in a
  // warp.
  DS_Max_Worker_Warp_Size_Log2 = 5,
  DS_Max_Worker_Warp_Size_Log2_Mask = (~0u >> (32-DS_Max_Worker_Warp_Size_Log2)),
  // The slot size that should be reserved for a working warp.
  DS_Worker_Warp_Slot_Size = DS_Max_Worker_Warp_Size * DS_Slot_Size,
};

} // namespace

// \brief Return the address where the parallelism level is kept in shared
// memory for the current thread. It is assumed we have up to 992 parallel
// worker threads.
// FIXME: Make this value reside in a descriptor whose size is decided at
// runtime (extern shared memory). This can be used for the other thread
// specific state as well.
LValue
CGOpenMPRuntimeNVPTX::getParallelismLevelLValue(CodeGenFunction &CGF) const {
  auto &M = CGM.getModule();

  const char *Name = "__openmp_nvptx_parallelism_levels";
  llvm::GlobalVariable *Gbl = M.getGlobalVariable(Name);

  if (!Gbl) {
    auto *Ty = llvm::ArrayType::get(CGM.Int32Ty, DS_Max_Worker_Threads);
    Gbl = new llvm::GlobalVariable(
        M, Ty,
        /*isConstant=*/false, llvm::GlobalVariable::CommonLinkage,
        llvm::Constant::getNullValue(Ty), Name,
        /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal,
        ADDRESS_SPACE_SHARED);
  }

  llvm::Value *Idx[] = {llvm::Constant::getNullValue(CGM.Int32Ty),
                        getNVPTXThreadID(CGF)};
  llvm::Value *AddrVal = CGF.Builder.CreateInBoundsGEP(Gbl, Idx);
  return CGF.MakeNaturalAlignAddrLValue(
      AddrVal, CGF.getContext().getIntTypeForBitwidth(/*DestWidth=*/32,
                                                      /*isSigned=*/true));
}

// \brief Return an integer with the parallelism level. Zero means that the
// current region is not enclosed in a parallel/simd region. The current level
// is kept in a shared memory array.
llvm::Value *
CGOpenMPRuntimeNVPTX::getParallelismLevel(CodeGenFunction &CGF) const {
  auto Addr = getParallelismLevelLValue(CGF);
  return CGF.EmitLoadOfLValue(Addr, SourceLocation()).getScalarVal();
}

// \brief Increase the value of parallelism level for the current thread.
void CGOpenMPRuntimeNVPTX::increaseParallelismLevel(
    CodeGenFunction &CGF) const {
  auto Addr = getParallelismLevelLValue(CGF);
  auto *CurVal = CGF.EmitLoadOfLValue(Addr, SourceLocation()).getScalarVal();
  auto *NewVal = CGF.Builder.CreateNSWAdd(CurVal, CGF.Builder.getInt32(1));
  CGF.EmitStoreOfScalar(NewVal, Addr);
}

// \brief Decrease the value of parallelism level for the current thread.
void CGOpenMPRuntimeNVPTX::decreaseParallelismLevel(
    CodeGenFunction &CGF) const {
  auto Addr = getParallelismLevelLValue(CGF);
  auto *CurVal = CGF.EmitLoadOfLValue(Addr, SourceLocation()).getScalarVal();
  auto *NewVal = CGF.Builder.CreateNSWSub(CurVal, CGF.Builder.getInt32(1));
  CGF.EmitStoreOfScalar(NewVal, Addr);
}

// \brief Initialize with zero the value of parallelism level for the current
// thread.
void CGOpenMPRuntimeNVPTX::initializeParallelismLevel(
    CodeGenFunction &CGF) const {
  auto Addr = getParallelismLevelLValue(CGF);
  CGF.EmitStoreOfScalar(llvm::Constant::getNullValue(CGM.Int32Ty), Addr);
}

static FieldDecl *addFieldToRecordDecl(ASTContext &C, DeclContext *DC,
                                       QualType FieldTy) {
  auto *Field = FieldDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Id=*/nullptr, FieldTy,
      C.getTrivialTypeSourceInfo(FieldTy, SourceLocation()),
      /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
  Field->setAccess(AS_public);
  DC->addDecl(Field);
  return Field;
}

// \brief Type of the data sharing master slot.
QualType
CGOpenMPRuntimeNVPTX::getDataSharingMasterSlotQty() {
  //  struct MasterSlot {
  //    Slot *Next;
  //    void *DataEnd;
  //    char Data[DS_Slot_Size]);
  //  };

  const char *Name = "__openmp_nvptx_data_sharing_master_slot_ty";
  if (DataSharingMasterSlotQty.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord(Name);
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, C.getPointerType(getDataSharingSlotQty()));
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()), DS_Slot_Size);
    QualType DataTy = C.getConstantArrayType(
        C.CharTy, NumElems, ArrayType::Normal, /*IndexTypeQuals=*/0);
    addFieldToRecordDecl(C, RD, DataTy);
    RD->completeDefinition();
    DataSharingMasterSlotQty = C.getRecordType(RD);
  }
  return DataSharingMasterSlotQty;
}

// \brief Type of the data sharing worker warp slot.
QualType
CGOpenMPRuntimeNVPTX::getDataSharingWorkerWarpSlotQty() {
  //  struct WorkerWarpSlot {
  //    Slot *Next;
  //    void *DataEnd;
  //    char [DS_Worker_Warp_Slot_Size];
  //  };

  const char *Name = "__openmp_nvptx_data_sharing_worker_warp_slot_ty";
  if (DataSharingWorkerWarpSlotQty.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord(Name);
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, C.getPointerType(getDataSharingSlotQty()));
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()),
                         DS_Worker_Warp_Slot_Size);
    QualType DataTy = C.getConstantArrayType(
        C.CharTy, NumElems, ArrayType::Normal, /*IndexTypeQuals=*/0);
    addFieldToRecordDecl(C, RD, DataTy);
    RD->completeDefinition();
    DataSharingWorkerWarpSlotQty = C.getRecordType(RD);
  }
  return DataSharingWorkerWarpSlotQty;
}

// \brief Get the type of the master or worker slot.
QualType CGOpenMPRuntimeNVPTX::getDataSharingSlotQty(bool UseFixedDataSize, bool IsMaster) {
  if (UseFixedDataSize) {
    if (IsMaster)
      return getDataSharingMasterSlotQty();
    return getDataSharingWorkerWarpSlotQty();
  }

  //  struct Slot {
  //    Slot *Next;
  //    void *DataEnd;
  //    char Data[];
  //  };

  const char *Name = "__kmpc_data_sharing_slot";
  if (DataSharingSlotQty.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord(Name);
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, C.getPointerType(C.getRecordType(RD)));
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    QualType DataTy = C.getIncompleteArrayType(C.CharTy, ArrayType::Normal,
                                               /*IndexTypeQuals=*/0);
    addFieldToRecordDecl(C, RD, DataTy);
    RD->completeDefinition();
    DataSharingSlotQty = C.getRecordType(RD);
  }
  return DataSharingSlotQty;
}

llvm::Type* CGOpenMPRuntimeNVPTX::getDataSharingSlotTy(bool UseFixedDataSize, bool IsMaster){
  return CGM.getTypes().ConvertTypeForMem(getDataSharingSlotQty(UseFixedDataSize, IsMaster));
}

// \brief Type of the data sharing root slot.
QualType CGOpenMPRuntimeNVPTX::getDataSharingRootSlotQty() {
  // The type of the global with the root slots:
  //  struct Slots {
  //    MasterSlot MS;
  //    WorkerWarpSlot WS[DS_Max_Worker_Threads/DS_Max_Worker_Warp_Size];
  // };
  if (DataSharingRootSlotQty.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord("__openmp_nvptx_data_sharing_ty");
    RD->startDefinition();
    addFieldToRecordDecl(C, RD,
                         getDataSharingMasterSlotQty());
    llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()),
                         DS_Max_Worker_Threads / DS_Max_Worker_Warp_Size);
    addFieldToRecordDecl(
        C, RD, C.getConstantArrayType(
                   getDataSharingWorkerWarpSlotQty(),
                   NumElems, ArrayType::Normal, /*IndexTypeQuals=*/0));
    RD->completeDefinition();
    DataSharingRootSlotQty = C.getRecordType(RD);
  }
  return DataSharingRootSlotQty;
}

// \brief Return address of the initial slot that is used to share data.
LValue CGOpenMPRuntimeNVPTX::getDataSharingRootSlotLValue(CodeGenFunction &CGF,
                                                         bool IsMaster) {
  auto &M = CGM.getModule();

  const char *Name = "__openmp_nvptx_shared_data_slots";
  llvm::GlobalVariable *Gbl = M.getGlobalVariable(Name);

  if (!Gbl) {
    auto *Ty = CGF.getTypes().ConvertTypeForMem(getDataSharingRootSlotQty());
    Gbl = new llvm::GlobalVariable(
        M, Ty,
        /*isConstant=*/false, llvm::GlobalVariable::CommonLinkage,
        llvm::Constant::getNullValue(Ty), Name,
        /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal);
  }

  // Return the master slot if the flag is set, otherwise get the right worker
  // slots.
  if (IsMaster) {
    llvm::Value *Idx[] = {llvm::Constant::getNullValue(CGM.Int32Ty),
                          llvm::Constant::getNullValue(CGM.Int32Ty)};
    llvm::Value *AddrVal = CGF.Builder.CreateInBoundsGEP(Gbl, Idx);
    return CGF.MakeNaturalAlignAddrLValue(
        AddrVal, getDataSharingMasterSlotQty());
  }

  auto *WarpID = getNVPTXWarpID(CGF);
  llvm::Value *Idx[] = {llvm::Constant::getNullValue(CGM.Int32Ty),
                        /*WS=*/CGF.Builder.getInt32(1), WarpID};
  llvm::Value *AddrVal = CGF.Builder.CreateInBoundsGEP(Gbl, Idx);
  return CGF.MakeNaturalAlignAddrLValue(
      AddrVal, getDataSharingWorkerWarpSlotQty());
}

// \brief Return the address where the address of the current slot is stored.
LValue
CGOpenMPRuntimeNVPTX::getSharedDataSlotPointerAddrLValue(CodeGenFunction &CGF,
                                                         bool IsMaster) {
  auto &M = CGM.getModule();
  auto &C = CGM.getContext();

  if (IsMaster) {
    const char *Name = "__openmp_nvptx_shared_data_current_master_slot_pointer";
    auto QTy = C.getPointerType(getDataSharingSlotQty());
    llvm::GlobalVariable *Gbl = M.getGlobalVariable(Name);

    if (!Gbl) {
      auto *Ty = CGF.getTypes().ConvertTypeForMem(QTy);
      Gbl = new llvm::GlobalVariable(
          M, Ty,
          /*isConstant=*/false, llvm::GlobalVariable::CommonLinkage,
          llvm::Constant::getNullValue(Ty), Name,
          /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal,
          ADDRESS_SPACE_SHARED);
    }
    return CGF.MakeNaturalAlignAddrLValue(Gbl, QTy);
  }

  const char *Name =
      "__openmp_nvptx_shared_data_current_worker_warp_slot_pointers";
  auto QTy = C.getPointerType(getDataSharingSlotQty());
  llvm::GlobalVariable *Gbl = M.getGlobalVariable(Name);

  if (!Gbl) {
    auto *ElemTy = CGF.getTypes().ConvertTypeForMem(QTy);
    auto *Ty = llvm::ArrayType::get(ElemTy, DS_Max_Worker_Threads /
                                                DS_Max_Worker_Warp_Size);
    Gbl = new llvm::GlobalVariable(
        M, Ty,
        /*isConstant=*/false, llvm::GlobalVariable::CommonLinkage,
        llvm::Constant::getNullValue(Ty), Name,
        /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal,
        ADDRESS_SPACE_SHARED);
  }

  llvm::Value *Idx[] = {llvm::Constant::getNullValue(CGM.Int32Ty),
                        getNVPTXWarpID(CGF)};
  auto *AddrVal = CGF.Builder.CreateInBoundsGEP(Gbl, Idx);
  return CGF.MakeNaturalAlignAddrLValue(AddrVal, QTy);
}

// \brief Return the address of the current data sharing slot.
LValue
CGOpenMPRuntimeNVPTX::getSharedDataSlotPointerLValue(CodeGenFunction &CGF,
                                                     bool IsMaster) {
  auto AddrLValue = getSharedDataSlotPointerAddrLValue(CGF, IsMaster);
  auto *Val = CGF.EmitLoadOfLValue(AddrLValue, SourceLocation()).getScalarVal();
  auto QTy = getDataSharingSlotQty(IsMaster);
  return CGF.MakeNaturalAlignAddrLValue(Val, QTy);
}

// \brief Return the address where the address of the current stack pointer (in
// the current slot) is stored.
LValue
CGOpenMPRuntimeNVPTX::getSharedDataStackPointerAddrLValue(CodeGenFunction &CGF,
                                                          bool IsMaster) {
  auto &M = CGM.getModule();
  auto &C = CGM.getContext();

  if (IsMaster) {
    const char *Name =
        "__openmp_nvptx_shared_data_current_master_stack_pointer";
    llvm::GlobalVariable *Gbl = M.getGlobalVariable(Name);

    if (!Gbl) {
      Gbl = new llvm::GlobalVariable(
          M, CGF.VoidPtrTy,
          /*isConstant=*/false, llvm::GlobalVariable::CommonLinkage,
          llvm::Constant::getNullValue(CGF.VoidPtrTy), Name,
          /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal,
          ADDRESS_SPACE_SHARED);
    }
    return CGF.MakeNaturalAlignAddrLValue(Gbl, C.VoidPtrTy);
  }

  const char *Name =
      "__openmp_nvptx_shared_data_current_worker_warp_stack_pointers";
  llvm::GlobalVariable *Gbl = M.getGlobalVariable(Name);

  if (!Gbl) {
    auto *ElemTy = CGF.VoidPtrTy;
    auto *Ty = llvm::ArrayType::get(ElemTy, DS_Max_Worker_Threads /
                                                DS_Max_Worker_Warp_Size);
    Gbl = new llvm::GlobalVariable(
        M, Ty,
        /*isConstant=*/false, llvm::GlobalVariable::CommonLinkage,
        llvm::Constant::getNullValue(Ty), Name,
        /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal,
        ADDRESS_SPACE_SHARED);
  }

  llvm::Value *Idx[] = {llvm::Constant::getNullValue(CGM.Int32Ty),
                        getNVPTXWarpID(CGF)};
  auto *AddrVal = CGF.Builder.CreateInBoundsGEP(Gbl, Idx);
  return CGF.MakeNaturalAlignAddrLValue(AddrVal, C.VoidPtrTy);
}

// \brief Return the address of the current data stack pointer.
LValue
CGOpenMPRuntimeNVPTX::getSharedDataStackPointerLValue(CodeGenFunction &CGF,
                                                      bool IsMaster) {
  auto AddrLValue = getSharedDataSlotPointerAddrLValue(CGF, IsMaster);
  auto *Val = CGF.EmitLoadOfLValue(AddrLValue, SourceLocation()).getScalarVal();
  return CGF.MakeNaturalAlignAddrLValue(Val, CGF.getContext().VoidTy);
}

// \brief Initialize the data sharing slots and pointers.
void CGOpenMPRuntimeNVPTX::initializeSharedData(CodeGenFunction &CGF,
                                                bool IsMaster) {
  // We initialized the slot and stack pointer in shared memory with their
  // initial values. Also, we initialize the slots with the initial size.

  auto &Bld = CGF.Builder;
  //auto &Ctx = CGF.getContext();

  // If this is not the OpenMP master thread, make sure that only the warp
  // master does the initialization.
  llvm::BasicBlock *EndBB = CGF.createBasicBlock("after_shared_data_init");
  ;
  if (!IsMaster) {
    auto *IsWarpMaster = getNVPTXIsWarpActiveMaster(CGF);
    llvm::BasicBlock *InitBB = CGF.createBasicBlock("shared_data_init");
    Bld.CreateCondBr(IsWarpMaster, InitBB, EndBB);
    CGF.EmitBlock(InitBB);
  }

  auto SlotLV = getDataSharingRootSlotLValue(CGF, IsMaster);

  // Initialize the slot and stack pointers.
  auto SlotPtrLV = getSharedDataSlotPointerAddrLValue(CGF, IsMaster);
  auto StackPtrLV = getSharedDataStackPointerAddrLValue(CGF, IsMaster);

  auto *SlotPtrTy = getDataSharingSlotTy()->getPointerTo();
  auto *CastedSlot =  Bld.CreateBitCast(SlotLV.getAddress(),SlotPtrTy).getPointer();

  llvm::Value *Args[] = {
      CastedSlot,
      SlotPtrLV.getPointer(),
      StackPtrLV.getPointer(),
      llvm::ConstantInt::get(CGM.SizeTy, IsMaster ? DS_Slot_Size : DS_Worker_Warp_Slot_Size) };
  Bld.CreateCall(createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_initialize_data_sharing_environment), Args);

//  auto SlotPtrQTy =
//      Ctx.getPointerType(getDataSharingSlotQty(IsMaster));
//  auto *SlotPtrTy = CGF.getTypes().ConvertTypeForMem(SlotPtrQTy);
//  auto SlotPtr = Bld.CreateBitCast(SlotLV.getAddress(), SlotPtrTy);
//  CGF.EmitStoreOfScalar(SlotPtr.getPointer(), SlotPtrLV);
//
//  auto StackPtrLV = getSharedDataStackPointerAddrLValue(CGF, IsMaster);
//  llvm::Value *Idx2[] = {Bld.getInt32(0), /*Data=*/Bld.getInt32(2),
//                         Bld.getInt32(0)};
//  auto *StackPtr = Bld.CreateInBoundsGEP(SlotLV.getPointer(), Idx2);
//  CGF.EmitStoreOfScalar(StackPtr, StackPtrLV);
//
//  // Initialize the DataEnd ( DataEnd = &Data[0] + Size).
//  auto *Base = Bld.CreatePtrToInt(StackPtr, CGF.IntPtrTy);
//  auto *Size = llvm::ConstantInt::get(CGF.IntPtrTy, IsMaster ? DS_Slot_Size
//          : DS_Worker_Warp_Slot_Size);
//  auto *DataEndVal = Bld.CreateNUWAdd(Base, Size);
//  DataEndVal = Bld.CreateIntToPtr(DataEndVal, CGF.VoidPtrTy);
//
//  llvm::Value *Idx[] = {llvm::Constant::getNullValue(CGM.Int32Ty),
//                        /*DataEnd=*/Bld.getInt32(1)};
//  auto *DataEndAddr = Bld.CreateGEP(SlotLV.getPointer(), Idx);
//  auto DataEndLV = CGF.MakeNaturalAlignAddrLValue(DataEndAddr, Ctx.VoidPtrTy);
//  CGF.EmitStoreOfScalar(DataEndVal,DataEndLV);

  CGF.EmitBlock(EndBB);
}

/// \brief Get the GPU warp size.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXWarpSize(CodeGenFunction &CGF) const {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(
      llvm::Intrinsic::getDeclaration(
          &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize),
      llvm::None, "nvptx_warp_size");
}

/// \brief Get the id of the current thread on the GPU.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXThreadID(CodeGenFunction &CGF) const {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(
      llvm::Intrinsic::getDeclaration(
          &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x),
      llvm::None, "nvptx_tid");
}

/// \brief Get the id of the current block on the GPU.
llvm::Value *CGOpenMPRuntimeNVPTX::getNVPTXBlockID(CodeGenFunction &CGF) const {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(
      llvm::Intrinsic::getDeclaration(
          &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x),
      llvm::None, "nvptx_block_id");
}

/// \brief Get the id of the warp in the block.
llvm::Value *CGOpenMPRuntimeNVPTX::getNVPTXWarpID(CodeGenFunction &CGF) const {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateAShr(getNVPTXThreadID(CGF), DS_Max_Worker_Warp_Size_Log2,
                        "nvptx_warp_id");
}

// \brief Get the maximum number of threads in a block of the GPU.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXNumThreads(CodeGenFunction &CGF) const {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(
      llvm::Intrinsic::getDeclaration(
          &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x),
      llvm::None, "nvptx_num_threads");
}

// \brief Get a 32 bit mask, whose bits set to 1 represent the active threads.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXWarpActiveThreadsMask(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_warp_active_thread_mask),
      None, "warp_active_thread_mask");
}

// \brief Get the number of active threads in a warp.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXWarpActiveNumThreads(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(llvm::Intrinsic::getDeclaration(
                            &CGM.getModule(), llvm::Intrinsic::nvvm_popc_i),
                        getNVPTXWarpActiveThreadsMask(CGF),
                        "warp_active_num_threads");
}

// \brief Get the ID of the thread among the current active threads in the warp.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXWarpActiveThreadID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;

  // The active thread Id can be computed as the number of bits in the active
  // mask to the right of the current thread:
  // popc( Mask << (32 - (threadID & 0x1f)) );
  auto *WarpID = Bld.CreateAnd(getNVPTXThreadID(CGF), Bld.getInt32(DS_Max_Worker_Warp_Size_Log2_Mask));
  auto *Mask = getNVPTXWarpActiveThreadsMask(CGF);
  auto *ShNum = Bld.CreateSub(Bld.getInt32(32), WarpID);
  auto *Sh = Bld.CreateShl(Mask, ShNum);
  return Bld.CreateCall(llvm::Intrinsic::getDeclaration(
                            &CGM.getModule(), llvm::Intrinsic::nvvm_popc_i),
                        Sh, "warp_active_thread_id");
}

// \brief Get a conditional that is set to true if the thread is the master of
// the active threads in the warp.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXIsWarpActiveMaster(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateICmpEQ(getNVPTXWarpActiveThreadID(CGF), Bld.getInt32(0),
                          "is_warp_active_master");
}

/// \brief Get barrier to synchronize all threads in a block.
void CGOpenMPRuntimeNVPTX::getNVPTXCTABarrier(CodeGenFunction &CGF) const {
  CGBuilderTy &Bld = CGF.Builder;
  Bld.CreateCall(llvm::Intrinsic::getDeclaration(
      &CGM.getModule(), llvm::Intrinsic::nvvm_barrier0));
}

/// \brief Get barrier #n to synchronize selected (multiple of 32) threads in
/// a block.
void CGOpenMPRuntimeNVPTX::getNVPTXBarrier(CodeGenFunction &CGF, int ID,
                                           int NumThreads) const {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Value *Args[] = {Bld.getInt32(ID), Bld.getInt32(NumThreads)};
  Bld.CreateCall(llvm::Intrinsic::getDeclaration(&CGM.getModule(),
                                                 llvm::Intrinsic::nvvm_barrier),
                 Args);
}

// \brief Synchronize all GPU threads in a block.
void CGOpenMPRuntimeNVPTX::syncCTAThreads(CodeGenFunction &CGF) const {
  getNVPTXCTABarrier(CGF);
}

//// \brief Emit code that allocates a memory chunk in global memory with size \a Size.
//llvm::Value *CGOpenMPRuntimeNVPTX::emitMallocCall(CodeGenFunction &CGF, QualType DataTy, llvm::Value *Size) {
//  CGBuilderTy &Bld = CGF.Builder;
//  auto *Ptr = CGF.EmitRuntimeCall(createNVPTXRuntimeFunction(OMPRTL_NVPTX__malloc), Size, "malloc_ptr");
//  auto *Ty = CGF.getTypes().ConvertTypeForMem(CGF.getContext().getPointerType(DataTy));
//  return Bld.CreateBitOrPointerCast(Ptr, Ty);
//}
//
//// \brief Deallocates the memory chunk pointed by \a Ptr;
//void CGOpenMPRuntimeNVPTX::emitFreeCall(CodeGenFunction &CGF, llvm::Value *Ptr){
//  CGBuilderTy &Bld = CGF.Builder;
//  Ptr = Bld.CreateBitOrPointerCast(Ptr, CGF.VoidPtrTy);
//  CGF.EmitRuntimeCall(createNVPTXRuntimeFunction(OMPRTL_NVPTX__free), Ptr);
//}

/// \brief Get the thread id of the OMP master thread.
/// The master thread id is the first thread (lane) of the last warp in the
/// GPU block.  Warp size is assumed to be some power of 2.
/// Thread id is 0 indexed.
/// E.g: If NumThreads is 33, master id is 32.
///      If NumThreads is 64, master id is 32.
///      If NumThreads is 1024, master id is 992.
llvm::Value *CGOpenMPRuntimeNVPTX::getMasterThreadID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Value *NumThreads = getNVPTXNumThreads(CGF);

  // We assume that the warp size is a power of 2.
  llvm::Value *Mask = Bld.CreateSub(getNVPTXWarpSize(CGF), Bld.getInt32(1));

  return Bld.CreateAnd(Bld.CreateSub(NumThreads, Bld.getInt32(1)),
                       Bld.CreateNot(Mask), "master_tid");
}

/// \brief Get number of OMP workers for parallel region after subtracting
/// the master warp.
llvm::Value *CGOpenMPRuntimeNVPTX::getNumWorkers(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateAdd(getMasterThreadID(CGF), Bld.getInt32(0), "num_workers");
}

/// \brief Get thread id in team.
/// FIXME: Remove the expensive remainder operation.
llvm::Value *CGOpenMPRuntimeNVPTX::getTeamThreadId(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateURem(getNVPTXThreadID(CGF), getMasterThreadID(CGF),
                        "team_tid");
}

/// \brief Get global thread id.
llvm::Value *CGOpenMPRuntimeNVPTX::getGlobalThreadId(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateAdd(Bld.CreateMul(getNVPTXBlockID(CGF), getNumWorkers(CGF)),
                       getTeamThreadId(CGF), "global_tid");
}

CGOpenMPRuntimeNVPTX::WorkerFunctionState::WorkerFunctionState(
    CodeGenModule &CGM)
    : WorkerFn(nullptr), CGFI(nullptr) {
  createWorkerFunction(CGM);
};

void CGOpenMPRuntimeNVPTX::WorkerFunctionState::createWorkerFunction(
    CodeGenModule &CGM) {
  // Create an worker function with no arguments.
  CGFI = &CGM.getTypes().arrangeNullaryFunction();

  WorkerFn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(*CGFI), llvm::GlobalValue::InternalLinkage,
      /* placeholder */ "_worker", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, WorkerFn, *CGFI);
  WorkerFn->setLinkage(llvm::GlobalValue::InternalLinkage);
  WorkerFn->addFnAttr(llvm::Attribute::NoInline);
}

void CGOpenMPRuntimeNVPTX::emitWorkerFunction(WorkerFunctionState &WST) {
  auto &Ctx = CGM.getContext();

  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
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
  syncCTAThreads(CGF);

  Address WorkFn = CGF.CreateTempAlloca(
      CGF.Int8PtrTy, CharUnits::fromQuantity(8), /*Name*/ "work_fn");
  Address WorkArgs = CGF.CreateTempAlloca(
      CGF.Int8PtrTy, CharUnits::fromQuantity(8), /*Name*/ "work_args");
  llvm::Value *Args[] = {WorkFn.getPointer(), WorkArgs.getPointer()};
  llvm::Value *IsActive = CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_parallel), Args,
      "is_active");

  // On termination condition (workfn == 0), exit loop.
  llvm::Value *ShouldTerminate = Bld.CreateICmpEQ(
      Bld.CreateLoad(WorkFn), llvm::Constant::getNullValue(CGF.Int8PtrTy),
      "should_terminate");
  Bld.CreateCondBr(ShouldTerminate, ExitBB, SelectWorkersBB);

  // Activate requested workers.
  CGF.EmitBlock(SelectWorkersBB);
  Bld.CreateCondBr(IsActive, ExecuteBB, BarrierBB);

  // Signal start of parallel region.
  CGF.EmitBlock(ExecuteBB);

  // Process work items: outlined parallel functions.
  for (auto *W : Work) {
    // Try to match this outlined function.
    auto ID = Bld.CreatePtrToInt(W, CGM.Int64Ty);
    ID = Bld.CreateIntToPtr(ID, CGM.Int8PtrTy);
    llvm::Value *WorkFnMatch =
        Bld.CreateICmpEQ(Bld.CreateLoad(WorkFn), ID, "work_match");

    llvm::BasicBlock *ExecuteFNBB = CGF.createBasicBlock(".execute.fn");
    llvm::BasicBlock *CheckNextBB = CGF.createBasicBlock(".check.next");
    Bld.CreateCondBr(WorkFnMatch, ExecuteFNBB, CheckNextBB);

    // Execute this outlined function.
    CGF.EmitBlock(ExecuteFNBB);

    //
    //
    // Get the arguments from block master to worker.
    //
    //

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
    auto Capture = Bld.CreateBitCast(Bld.CreateLoad(WorkArgs), CaptureType);
    FnArgs.push_back(Capture);

    // Insert call to work function.
    CGF.EmitCallOrInvoke(Fn, FnArgs);
    // Go to end of parallel region.
    CGF.EmitBranch(TerminateBB);

    CGF.EmitBlock(CheckNextBB);
  }

  // Signal end of parallel region.
  CGF.EmitBlock(TerminateBB);
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_end_parallel),
      ArrayRef<llvm::Value *>());
  CGF.EmitBranch(BarrierBB);

  // All active and inactive workers wait at a barrier after parallel region.
  CGF.EmitBlock(BarrierBB);
  // Barrier after parallel region.
  syncCTAThreads(CGF);
  CGF.EmitBranch(AwaitBB);

  // Exit target region.
  CGF.EmitBlock(ExitBB);
}

// Setup NVPTX threads for master-worker OpenMP scheme.
void CGOpenMPRuntimeNVPTX::emitEntryHeader(CodeGenFunction &CGF,
                                           EntryFunctionState &EST,
                                           WorkerFunctionState &WST) {
  CGBuilderTy &Bld = CGF.Builder;

  // Get the master thread id.
  llvm::Value *MasterID = getMasterThreadID(CGF);
  // Current thread's identifier.
  llvm::Value *ThreadID = getNVPTXThreadID(CGF);

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
  initializeParallelismLevel(CGF);
  //initializeSharedData(CGF, /*IsMaster=*/false);
  llvm::SmallVector<llvm::Value *, 16> WorkerVars;
  for (auto &I : CGF.CurFn->args()) {
    WorkerVars.push_back(&I);
  }

  CGF.EmitCallOrInvoke(WST.WorkerFn, llvm::None);
  CGF.EmitBranch(EST.ExitBB);

  // Only master thread executes subsequent serial code.
  CGF.EmitBlock(MasterBB);
  //initializeSharedData(CGF, /*IsMaster=*/true);

  // First action in sequential region:
  // Initialize the state of the OpenMP runtime library on the GPU.
  llvm::Value *Args[] = {Bld.getInt32(/*OmpHandle=*/0), getNVPTXThreadID(CGF)};
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_init), Args);
}

void CGOpenMPRuntimeNVPTX::emitEntryFooter(CodeGenFunction &CGF,
                                           EntryFunctionState &EST) {
  llvm::BasicBlock *TerminateBB = CGF.createBasicBlock(".termination.notifier");
  CGF.EmitBranch(TerminateBB);

  CGF.EmitBlock(TerminateBB);
  // Signal termination condition.
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_deinit), None);
  // Barrier to terminate worker threads.
  syncCTAThreads(CGF);
  // Master thread jumps to exit point.
  CGF.EmitBranch(EST.ExitBB);

  CGF.EmitBlock(EST.ExitBB);
}

/// \brief Returns specified OpenMP runtime function for the current OpenMP
/// implementation.  Specialized for the NVPTX device.
/// \param Function OpenMP runtime function.
/// \return Specified function.
llvm::Constant *
CGOpenMPRuntimeNVPTX::createNVPTXRuntimeFunction(unsigned Function) {
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
  case OMPRTL_NVPTX__kmpc_kernel_deinit: {
    // Build void __kmpc_kernel_deinit();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, {}, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_deinit");
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
    /// void *outlined_function, void **args, kmp_int32 nArgs);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy, CGM.Int8PtrPtrTy, CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_prepare_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_parallel: {
    /// Build bool __kmpc_kernel_parallel(
    /// void **outlined_function, void **args);
    llvm::Type *TypeParams[] = {CGM.Int8PtrPtrTy, CGM.Int8PtrPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(llvm::Type::getInt1Ty(CGM.getLLVMContext()),
                                TypeParams, /*isVarArg*/ false);
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
  case OMPRTL_NVPTX__kmpc_kernel_convergent_parallel: {
    /// \brief Call to bool __kmpc_kernel_convergent_parallel(
    /// bool *IsFinal, kmpc_int32 *LaneSource);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy, CGM.Int8PtrTy,
                                CGM.Int32Ty->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(llvm::Type::getInt1Ty(CGM.getLLVMContext()),
                                TypeParams, /*isVarArg*/ false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_convergent_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_end_convergent_parallel: {
    /// Build void __kmpc_kernel_end_convergent_parallel();
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy,
                                      "__kmpc_kernel_end_convergent_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_warp_active_thread_mask: {
    /// Build void __kmpc_warp_active_thread_mask();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, None, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_warp_active_thread_mask");
    break;
  }
//  case OMPRTL_NVPTX__malloc: {
//    /// Build void * malloc(size_t size);
//    llvm::Type *TypeParams[] = {CGM.SizeTy};
//    llvm::FunctionType *FnTy =
//        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg*/ false);
//    RTLFn = CGM.CreateRuntimeFunction(FnTy, "malloc");
//    break;
//  }
//  case OMPRTL_NVPTX__free: {
//    /// Build void free(void *ptr);
//    llvm::Type *TypeParams[] = {CGM.VoidPtrTy};
//    llvm::FunctionType *FnTy =
//        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
//    RTLFn = CGM.CreateRuntimeFunction(FnTy, "free");
//    break;
//  }

  case OMPRTL_NVPTX__kmpc_initialize_data_sharing_environment: {
    /// Build void __kmpc_initialize_data_sharing_environment(__kmpc_data_sharing_slot *RootS, __kmpc_data_sharing_slot **SharedS, void **SharedD, size_t InitialDataSize);
    auto *SlotTy = CGM.getTypes().ConvertTypeForMem(getDataSharingSlotQty());
    llvm::Type *TypeParams[] = {
        SlotTy->getPointerTo(),
        SlotTy->getPointerTo()->getPointerTo(ADDRESS_SPACE_SHARED),
        CGM.VoidPtrTy->getPointerTo(ADDRESS_SPACE_SHARED),
        CGM.SizeTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_initialize_data_sharing_environment");
    break;
  }
  case OMPRTL_NVPTX__kmpc_data_sharing_environment_begin: {
    /// Build void* __kmpc_data_sharing_environment_begin(__kmpc_data_sharing_slot **SharedS, void **SharedD, __kmpc_data_sharing_slot **SavedSharedS, void **SavedSharedD, size_t SharingDataSize, size_t SharingDefaultDataSize, int32_t *ReuseData);

    auto *SlotTy = CGM.getTypes().ConvertTypeForMem(getDataSharingSlotQty());
    llvm::Type *TypeParams[] = {
        SlotTy->getPointerTo()->getPointerTo(ADDRESS_SPACE_SHARED),
        CGM.VoidPtrTy->getPointerTo(ADDRESS_SPACE_SHARED),
        SlotTy->getPointerTo()->getPointerTo(),
        CGM.VoidPtrTy->getPointerTo(),
        CGM.SizeTy,
        CGM.SizeTy,
        CGM.Int32Ty->getPointerTo()
    };
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_data_sharing_environment_begin");
    break;
  }
  case OMPRTL_NVPTX__kmpc_data_sharing_environment_end: {
    /// Build void __kmpc_data_sharing_environment_end( __kmpc_data_sharing_slot **SharedS, void **SharedD, __kmpc_data_sharing_slot **SavedSharedS, void **SavedSharedD);
    auto *SlotTy = CGM.getTypes().ConvertTypeForMem(getDataSharingSlotQty());
    llvm::Type *TypeParams[] = {
        SlotTy->getPointerTo()->getPointerTo(ADDRESS_SPACE_SHARED),
        CGM.VoidPtrTy->getPointerTo(ADDRESS_SPACE_SHARED),
        SlotTy->getPointerTo()->getPointerTo(),
        CGM.VoidPtrTy->getPointerTo(),
    };
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_data_sharing_environment_end");
    break;
  }
  }
  return RTLFn;
}

llvm::Value *CGOpenMPRuntimeNVPTX::getThreadID(CodeGenFunction &CGF,
                                               SourceLocation Loc) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");
  return getGlobalThreadId(CGF);
}

void CGOpenMPRuntimeNVPTX::emitCapturedVars(
    CodeGenFunction &CGF, const OMPExecutableDirective &S,
    llvm::SmallVector<llvm::Value *, 16> &CapturedVars) {
  auto CS = cast<CapturedStmt>(S.getAssociatedStmt());
  auto Var = CGF.GenerateCapturedStmtArgument(*CS);
  CapturedVars.push_back(Var.getPointer());
}

void CGOpenMPRuntimeNVPTX::createOffloadEntry(llvm::Constant *ID,
                                              llvm::Constant *Addr,
                                              uint64_t Size) {
  auto *F = dyn_cast<llvm::Function>(Addr);
  // TODO: Add support for global variables on the device after declare target
  // support.
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
}

void CGOpenMPRuntimeNVPTX::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry) {
  if (!IsOffloadEntry) // Nothing to do.
    return;

  assert(!ParentName.empty() && "Invalid target region parent name!");

  const CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());

  EntryFunctionState EST;
  WorkerFunctionState WST(CGM);

  // Emit target region as a standalone region.
  auto &&CodeGen = [&EST, &WST, &CS, this, &D](CodeGenFunction &CGF) {
    emitEntryHeader(CGF, EST, WST);
    CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
    CGF.EmitOMPPrivateClause(D, PrivateScope);
    (void)PrivateScope.Privatize();

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

  return;
}

// void CGOpenMPRuntimeNVPTX::enterTarget() {
//  IsOrphaned = false;
//  ParallelNestingLevel = 0;
//}
//
// void CGOpenMPRuntimeNVPTX::exitTarget() {
//  IsOrphaned = true;
//  ParallelNestingLevel = 0;
//  Work.clear();
//}

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

namespace {
class ParallelNestingLevelRAII {
private:
  int &ParallelNestingLevel;

public:
  ParallelNestingLevelRAII(int &ParallelNestingLevel)
      : ParallelNestingLevel(ParallelNestingLevel) {
    ParallelNestingLevel++;
  }
  ~ParallelNestingLevelRAII() { ParallelNestingLevel--; }
};
} // namespace

llvm::Value *CGOpenMPRuntimeNVPTX::emitParallelOrTeamsOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  assert(ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 *");

  llvm::Function *OutlinedFun = nullptr;
  if (isa<OMPTeamsDirective>(D)) {
    // no outlining happening for teams
  } else {
    const CapturedStmt *CS = cast<CapturedStmt>(D.getAssociatedStmt());
    CodeGenFunction CGF(CGM, true);
    bool HasCancel = false;
    if (auto *OPD = dyn_cast<OMPParallelDirective>(&D))
      HasCancel = OPD->hasCancel();
    else if (auto *OPSD = dyn_cast<OMPParallelSectionsDirective>(&D))
      HasCancel = OPSD->hasCancel();
    else if (auto *OPFD = dyn_cast<OMPParallelForDirective>(&D))
      HasCancel = OPFD->hasCancel();

    // Include updates in runtime parallelism level.
    auto &&CodeGenWithDataSharing = [this, &CodeGen](CodeGenFunction &CGF) {
      increaseParallelismLevel(CGF);
      CodeGen(CGF);
      decreaseParallelismLevel(CGF);
    };

    CGOpenMPOutlinedRegionInfo CGInfo(*CS, ThreadIDVar, CodeGenWithDataSharing,
                                      InnermostKind, HasCancel);
    CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
    ParallelNestingLevelRAII NestingRAII(ParallelNestingLevel);
    // The outlined function takes as arguments the global_tid, bound_tid,
    // and a capture structure created from the captured variables.
    OutlinedFun = CGF.GenerateCapturedStmtFunction(*CS);
  }
  return OutlinedFun;
}

bool CGOpenMPRuntimeNVPTX::InL0() {
  return !IsOrphaned && ParallelNestingLevel == 0;
}

bool CGOpenMPRuntimeNVPTX::InL1() {
  return !IsOrphaned && ParallelNestingLevel == 1;
}

bool CGOpenMPRuntimeNVPTX::InL1Plus() {
  return !IsOrphaned && ParallelNestingLevel >= 1;
}

bool CGOpenMPRuntimeNVPTX::IndeterminateLevel() { return IsOrphaned; }

// \brief Obtain the data sharing info for the current context.
const CGOpenMPRuntimeNVPTX::DataSharingInfo &CGOpenMPRuntimeNVPTX::getDataSharingInfo(CodeGenFunction &CGF){
  auto *Context = CGF.CurCodeDecl;
  assert(Context && "A parallel region is expected to be enclosed in a context.");

  ASTContext &C = CGM.getContext();

  auto It = DataSharingInfoMap.find(Context);

  // Check if the info was created before.
  if (It != DataSharingInfoMap.end())
    return It->second;

  auto &Info = DataSharingInfoMap[Context];

  // Get the body of the region. The region context is either a function or a captured declaration.
  const Stmt *Body;
  if (auto *D = dyn_cast<CapturedDecl>(Context))
    Body = D->getBody();
  else
    Body = cast<FunctionDecl>(D)->getBody();

  // Find all the captures in all enclosed regions and obtain their captured statements.
  SmallVector<const CapturedStmt*, 8> CapturedStmts;
  SmallVector<const Stmt*, 64> WorkList;
  WorkList.push_back(Body);
  while(!WorkList.empty()) {
    const Stmt *CurStmt = WorkList.pop_back_val();
    if (!CurStmt)
      continue;

    // Is this a parallel region.
    if (auto *Dir = dyn_cast<OMPExecutableDirective>(CurStmt))
      if (isOpenMPParallelDirective(Dir->getDirectiveKind())) {
        CapturedStmts.push_back(cast<CapturedStmt>(Dir->getAssociatedStmt()));
        continue;
      }

    // Keep looking for other regions.
    WorkList.append(CurStmt->child_begin(),CurStmt->child_end());
  }

  assert(!CapturedStmts.empty() && "Expecting at least one parallel region!");

  // Scan the captured statements and generate a record to contain all the data to be shared. Make sure we do not share the same thing twice.
  auto *SharedMasterRD = C.buildImplicitRecord("__openmp_nvptx_data_sharing_master_record");
  auto *SharedWarpRD = C.buildImplicitRecord("__openmp_nvptx_data_sharing_warp_record");
  SharedMasterRD->startDefinition();
  SharedWarpRD->startDefinition();

  llvm::SmallSet<const Decl*, 32> AlreadySharedDecls;
  for (auto *CS : CapturedStmts) {
    const RecordDecl *RD = CS->getCapturedRecordDecl();
    auto CurField = RD->field_begin();
    auto CurCap = CS->captures().begin();
    for (CapturedStmt::const_capture_init_iterator I = CS->capture_init_begin(),
                                                   E = CS->capture_init_end();
         I != E; ++I, ++CurField, ++CurCap) {

      // Skip if it was already shared.
      if (AlreadySharedDecls.count(*CurField))
        continue;

      AlreadySharedDecls.insert(*CurField);

      QualType ElemTy;
      if (CurField->hasCapturedVLAType()) {
        assert("VLAs are not yet supported in NVPTX target data sharing!");
        continue;
      } else if (CurCap->capturesThis()) {
        ElemTy = (*I)->getType();
        Info.CapturesValues.push_back(CGF.LoadCXXThis());
      } else if (CurCap->capturesVariableByCopy()) {
        assert("Not expecting to capture variables by copy in NVPTX target data sharing!");
        continue;
      } else {
        // Get the reference to the variable that is initializing the capture.
        const DeclRefExpr *DRE = cast<DeclRefExpr>(*I);
        const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
        assert(VD->hasLocalStorage() && "Expecting to capture only variables with local storage.");
        Info.CapturesValues.push_back(CGF.GetAddrOfLocalVar(VD).getPointer());
      }

      addFieldToRecordDecl(C, SharedMasterRD, ElemTy);
      llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()),DS_Max_Worker_Warp_Size);
      auto QTy = C.getConstantArrayType(ElemTy, NumElems, ArrayType::Normal, /*IndexTypeQuals=*/0);
      addFieldToRecordDecl(C, SharedWarpRD, QTy);
    }
  }

  SharedMasterRD->completeDefinition();
  SharedWarpRD->completeDefinition();
  Info.MasterRecordType = C.getRecordType(SharedMasterRD);
  Info.WorkerWarpRecordType = C.getRecordType(SharedWarpRD);

  return Info;
//  auto ShareRDTy = C.getRecordType(SharedRD);
//
//  // Now that we have a record type suitable to the data sharing, we need to check if we have room in the current slot of the stack. If not, we need to grow it. If this is not the master, only the warp master should grow the stack.
//
//  llvm::BasicBlock *EndBB = CGF.createBasicBlock("after_shared_stack_check");
//  if (!IsMaster) {
//    llvm::BasicBlock *WarpBB = CGF.createBasicBlock("warp_shared_stack_check");
//    auto *IsWarpMaster = getNVPTXIsWarpActiveMaster(CGF);
//    Bld.CreateCondBr(IsWarpMaster, WarpBB, EndBB);
//    CGF.EmitBlock(WarpBB);
//  }
//
//  // Save the current stack and slot pointer - this is requires to restore them after the region.
//  auto SavedSharedStackAddr = CGF.CreateMemTemp(C.VoidPtrTy, "saved_shared_stack_pointer");
//  auto SharedStackPointer = getSharedDataStackPointerLValue(CGF, IsMaster);
//  CGF.EmitStoreOfScalar(Bld.CreateBitCast(SharedStackPointer.getAddress(), CGM.VoidPtrTy).getPointer(), SavedSharedStackAddr, /*Volatile=*/false, C.VoidPtrTy);
//
//  QualType SharedSlotPtrQTy = C.getPointerType(getDataSharingSlotQty(IsMaster));
//  auto *SharedSlotPtrTy =  CGF.getTypes().ConvertTypeForMem(SharedSlotPtrQTy);
//
//  auto SavedSharedSlotAddr = CGF.CreateMemTemp(SharedSlotPtrQTy, "saved_shared_slot_pointer");
//  auto SharedSlotPointer = getSharedDataSlotPointerLValue(CGF, IsMaster);
//  CGF.EmitStoreOfScalar(Bld.CreateBitCast(SharedSlotPointer.getAddress(), SharedSlotPtrTy).getPointer(), SavedSharedSlotAddr, /*Volatile=*/false, SharedSlotPtrQTy);
//
//  // Clean up the 'Next' entry in the slot. If it is not NULL is that because some inner region had to grow the stack.
//  llvm::Value *Idx[] = { Bld.getInt32(0), /*Next=*/Bld.getInt32(0) };
//  auto *SlotNextAddr =  Bld.CreateInBoundsGEP(SharedSlotPointer.getPointer(), Idx);
//  auto SlotNextAddrLV = CGF.MakeNaturalAlignAddrLValue(SlotNextAddr, C.VoidPtrTy);
//  auto *SlotNext = CGF.EmitLoadOfScalar(SlotNextAddrLV, SourceLocation());
//  auto *RequiresCleanUp = Bld.CreateIsNotNull(SlotNext,"needs_stack_clean_up");
//
//  llvm::BasicBlock *CleanUpBB = CGF.createBasicBlock("stack_clean_up");
//  llvm::BasicBlock *AfterCleanUpBB = CGF.createBasicBlock("after_stack_clean_up");
//  Bld.CreateCondBr(RequiresCleanUp, CleanUpBB, AfterCleanUpBB);
//  CGF.EmitBlock(CleanUpBB);
//  emitFreeCall(CGF, SlotNext);
//  CGF.EmitBlock(AfterCleanUpBB);
//
//  // Get the end address of the slot.
//  llvm::Value *Idx[] = { Bld.getInt32(0), /*DataEnd=*/ Bld.getInt32(1) };
//  auto *SlotEndAddr =  Bld.CreateInBoundsGEP(SharedSlotPointer.getPointer(), Idx);
//  auto SlotEndAddrLV = CGF.MakeNaturalAlignAddrLValue(SlotEndAddr, C.VoidPtrTy);
//  auto *SlotEnd = CGF.EmitLoadOfScalar(SlotEndAddrLV, SourceLocation());
//  SlotEnd = Bld.CreatePtrToInt(SlotEnd, CGF.IntPtrTy);
//
//  // Add the size of the record to the current stack pointer and check if it can fit in the stack.
//  auto *SlotEndRequired = Bld.CreatePtrToInt(SharedStackPointer.getPointer(), CGF.IntPtrTy);
//
//  auto RecordSizeInBytes = C.getTypeSizeInChars(ShareRDTy).getQuantity();
//  SlotEndRequired = Bld.CreateNUWAdd(SlotEndRequired, llvm::ConstantInt::get(CGF.IntPtrTy, RecordSizeInBytes));
//  auto *NeedsToGrow = Bld.CreateICmpUGE(SlotEndRequired, SlotEnd, "needs_stack_grow");
//
//  auto *GrowBB = CGF.createBasicBlock("grow_shared_memory_stack");
//  auto *AfterGrowBB = CGF.createBasicBlock("after_grow_shared_memory_stack");
//  Bld.CreateCondBr(NeedsToGrow, GrowBB, AfterGrowBB);
//  CGF.EmitBlock(GrowBB);
//
//  // Growing the stack consists of the allocation of a new slot whose size is the maximum between the default and what is required in this capture environment.
//  auto NewSlotRequiredSize =  RecordSizeInBytes + /*Next*/C.getTypeSizeInChars(C.VoidPtrTy).getQuantity() + /*DataEnd*/;
//
//  CGF.EmitBlock(AfterGrowBB);
//
//  CGF.EmitBlock(EndBB);
}

const CGOpenMPRuntimeNVPTX::DataSharingInfo &CGOpenMPRuntimeNVPTX::getExistingDataSharingInfo(const Decl *Context){
  auto It = DataSharingInfoMap.find(Context);
  assert(It != DataSharingInfoMap.end() && "Data sharing info does not exist.");
  return It->second;
}

// \brief Emit the code that each thread requires to execute when it encounters
// one of the three possible parallelism level. This also emits the required
// data sharing code for each level.
void CGOpenMPRuntimeNVPTX::emitParallelismLevelCode(
    CodeGenFunction &CGF, const RegionCodeGenTy &Level0,
    const RegionCodeGenTy &Level1, const RegionCodeGenTy &Sequential) {
  auto &Bld = CGF.Builder;

  // Flags that prevent code to be emitted if it can be proven that threads
  // cannot reach this function at a given level.
  //
  // FIXME: This current relies on a simple analysis that may not be correct if
  // we have function in a target region.
  bool OnlyInL0 = InL0();
  bool OnlyInL1 = InL1();
  bool OnlySequential = !IsOrphaned && !InL0() && !InL1();

  // Emit runtime checks if we cannot prove this code is reached only at a
  // certain parallelism level.
  //
  // For each level i the code will look like:
  //
  //   isLevel = icmp Level, i;
  //   br isLevel, .leveli.parallel, .next.parallel
  //
  // .leveli.parallel:
  //   ; code for level i + shared data code
  //   br .after.parallel
  //
  // .next.parallel

  llvm::BasicBlock *AfterBB = CGF.createBasicBlock(".after.parallel");

  // Do we need to emit L0 code?
  if (!OnlyInL1 && !OnlySequential) {
    llvm::BasicBlock *LBB = CGF.createBasicBlock(".level0.parallel");
    llvm::BasicBlock *NextBB = CGF.createBasicBlock(".next.parallel");

    // Do we need runtime checks
    if (!OnlyInL0) {
      auto *ThreadID = getNVPTXThreadID(CGF);
      auto *MasterID = getMasterThreadID(CGF);
      auto *Cond = Bld.CreateICmpEQ(ThreadID, MasterID);
      Bld.CreateCondBr(Cond, LBB, NextBB);
    }

    CGF.EmitBlock(LBB);

    // Fill up captures of shared data here.
    //
    // Captures here will be a struct of arrays with a single element each (one
    // array per capture).

    Level0(CGF);

    // Free captures of shared data here.

    CGF.EmitBranch(AfterBB);
    CGF.EmitBlock(NextBB);
  }

  // Do we need to emit L1 code?
  if (!OnlyInL0 && !OnlySequential) {
    llvm::BasicBlock *LBB = CGF.createBasicBlock(".level1.parallel");
    llvm::BasicBlock *NextBB = CGF.createBasicBlock(".next.parallel");

    // Do we need runtime checks
    if (!OnlyInL1) {
      auto *ParallelLevelVal = getParallelismLevel(CGF);
      auto *Cond = Bld.CreateICmpEQ(ParallelLevelVal, Bld.getInt32(1));
      Bld.CreateCondBr(Cond, LBB, NextBB);
    }

    CGF.EmitBlock(LBB);

    // Fill up captures of shared data here.
    //
    // Captures here will be a struct of arrays with 32 elements. It is possible
    // some threads do not reach this point which will cause waste but that is
    // the only way to generate this statically.

    Level1(CGF);

    // Free captures of shared data here.

    CGF.EmitBranch(AfterBB);
    CGF.EmitBlock(NextBB);
  }

  // Do we need to emit sequential code?
  if (!OnlyInL0 && !OnlyInL1) {
    llvm::BasicBlock *SeqBB = CGF.createBasicBlock(".sequential.parallel");

    // Do we need runtime checks
    if (!OnlySequential) {
      auto *ParallelLevelVal = getParallelismLevel(CGF);
      auto *Cond = Bld.CreateICmpSGT(ParallelLevelVal, Bld.getInt32(1));
      Bld.CreateCondBr(Cond, SeqBB, AfterBB);
    }

    CGF.EmitBlock(SeqBB);
    Sequential(CGF);
  }

  CGF.EmitBlock(AfterBB);
}

void CGOpenMPRuntimeNVPTX::emitParallelCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> CapturedVars, const Expr *IfCond) {
  if (!CGF.HaveInsertPoint())
    return;
  llvm::Function *Fn = cast<llvm::Function>(OutlinedFn);
  // Force inline this outlined function at its call site.
  // Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  Fn->setLinkage(llvm::GlobalValue::InternalLinkage);

  //
  //
  // Emit code that does the changes in the beginning of the function.
  //
  //


  auto *RTLoc = emitUpdateLocation(CGF, Loc);
  auto &&L0ParallelGen = [this, Fn, &CapturedVars](CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;

    auto Capture = CapturedVars.front();
    auto CaptureType = Capture->getType()->getArrayElementType();
    auto ID = Bld.CreatePtrToInt(Fn, CGM.Int64Ty);
    ID = Bld.CreateIntToPtr(ID, CGM.Int8PtrTy);

    // Prepare for parallel region.  Indicate the outlined function and request
    // a buffer to place its arguments.
    Address WorkArgs = CGF.CreateTempAlloca(
        CGF.Int8PtrTy, CharUnits::fromQuantity(8), /*Name*/ "work_args");
    llvm::Value *Args[] = {
        ID, WorkArgs.getPointer(),
        /*nArgs=*/Bld.getInt32(CaptureType->getStructNumElements())};
    CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_prepare_parallel),
        Args);

    // Copy variables in the capture buffer to the shared workargs structure for
    // use by the workers.  The extraneous capture buffer will be optimized out
    // by llvm.
    auto WorkArgsPtr =
        Bld.CreateBitCast(Bld.CreateLoad(WorkArgs), Capture->getType());
    for (unsigned idx = 0; idx < CaptureType->getStructNumElements(); idx++) {
      auto Src = Bld.CreateConstInBoundsGEP2_32(CaptureType, Capture, 0, idx);
      auto Dst =
          Bld.CreateConstInBoundsGEP2_32(CaptureType, WorkArgsPtr, 0, idx);
      Bld.CreateDefaultAlignedStore(Bld.CreateDefaultAlignedLoad(Src), Dst);
    }

    // Activate workers.
    syncCTAThreads(CGF);

    // Barrier at end of parallel region.
    syncCTAThreads(CGF);

    // Remember for post-processing in worker loop.
    Work.push_back(Fn);
  };
  auto &&L1ParallelGen = [this, Fn, &CapturedVars, &RTLoc,
                          &Loc](CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    clang::ASTContext &Ctx = CGF.getContext();

    Address IsFinal =
        CGF.CreateTempAlloca(CGF.Int8Ty, CharUnits::fromQuantity(1),
                             /*Name*/ "is_final");
    Address WorkSource =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ "work_source");
    llvm::APInt TaskBufferSize(/*numBits=*/32, TASK_STATE_SIZE);
    auto TaskBufferTy = Ctx.getConstantArrayType(
        Ctx.CharTy, TaskBufferSize, ArrayType::Normal, /*IndexTypeQuals=*/0);
    auto TaskState = CGF.CreateMemTemp(TaskBufferTy, CharUnits::fromQuantity(8),
                                       /*Name=*/"task_state")
                         .getPointer();
    CGF.InitTempAlloca(IsFinal, Bld.getInt8(/*C=*/0));
    CGF.InitTempAlloca(WorkSource, Bld.getInt32(/*C=*/-1));

    llvm::BasicBlock *DoBodyBB = CGF.createBasicBlock(".do.body");
    llvm::BasicBlock *ExecuteBB = CGF.createBasicBlock(".do.body.execute");
    llvm::BasicBlock *DoCondBB = CGF.createBasicBlock(".do.cond");
    llvm::BasicBlock *DoEndBB = CGF.createBasicBlock(".do.end");

    CGF.EmitBranch(DoBodyBB);
    CGF.EmitBlock(DoBodyBB);
    auto ArrayDecay = Bld.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.Int8Ty, TASK_STATE_SIZE), TaskState,
        /*Idx0=*/0, /*Idx1=*/0);
    llvm::Value *Args[] = {ArrayDecay, IsFinal.getPointer(),
                           WorkSource.getPointer()};
    llvm::Value *IsActive =
        CGF.EmitRuntimeCall(createNVPTXRuntimeFunction(
                                OMPRTL_NVPTX__kmpc_kernel_convergent_parallel),
                            Args);
    Bld.CreateCondBr(IsActive, ExecuteBB, DoCondBB);

    CGF.EmitBlock(ExecuteBB);

    //
    //
    // Get the arguments from worker master to worker.
    //
    //

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
    ArrayDecay = Bld.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.Int8Ty, TASK_STATE_SIZE), TaskState,
        /*Idx0=*/0, /*Idx1=*/0);
    llvm::Value *EndArgs[] = {ArrayDecay};
    CGF.EmitRuntimeCall(createNVPTXRuntimeFunction(
                            OMPRTL_NVPTX__kmpc_kernel_end_convergent_parallel),
                        EndArgs);
    CGF.EmitBranch(DoCondBB);

    CGF.EmitBlock(DoCondBB);
    llvm::Value *IsDone = Bld.CreateICmpEQ(Bld.CreateLoad(IsFinal),
                                           Bld.getInt8(/*C=*/1), "is_done");
    Bld.CreateCondBr(IsDone, DoEndBB, DoBodyBB);

    CGF.EmitBlock(DoEndBB);
  };

  auto &&SeqGen = [this, Fn, &CapturedVars, &RTLoc,
                   &Loc](CodeGenFunction &CGF) {
    auto DL = CGM.getDataLayout();
    auto ThreadID = getThreadID(CGF, Loc);
    // Build calls:
    // __kmpc_serialized_parallel(&Loc, GTid);
    llvm::Value *Args[] = {RTLoc, ThreadID};
    CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_serialized_parallel),
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
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_end_serialized_parallel),
        EndArgs);
  };

  auto &&ThenGen = [this, &L0ParallelGen, &L1ParallelGen,
                    &SeqGen](CodeGenFunction &CGF) {
    emitParallelismLevelCode(CGF, L0ParallelGen, L1ParallelGen, SeqGen);
  };

  if (IfCond) {
    emitOMPIfClause(CGF, IfCond, ThenGen, SeqGen);
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

CGOpenMPRuntimeNVPTX::CGOpenMPRuntimeNVPTX(CodeGenModule &CGM)
    : CGOpenMPRuntime(CGM), IsOrphaned(false), ParallelNestingLevel(0) {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP NVPTX can only handle device code.");
}

void CGOpenMPRuntimeNVPTX::emitNumTeamsClause(CodeGenFunction &CGF,
                                              const Expr *NumTeams,
                                              const Expr *ThreadLimit,
                                              SourceLocation Loc) {}

void CGOpenMPRuntimeNVPTX::emitTeamsCall(CodeGenFunction &CGF,
                                    const OMPExecutableDirective &D,
                                    SourceLocation Loc,
                                    llvm::Value *OutlinedFn,
                                    ArrayRef<llvm::Value *> CapturedVars) {

  // just emit the statements in the teams region inlined
  auto &&CodeGen = [&D](CodeGenFunction &CGF) {
    CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
    (void)CGF.EmitOMPFirstprivateClause(D, PrivateScope);
    CGF.EmitOMPPrivateClause(D, PrivateScope);
    (void)PrivateScope.Privatize();

    CGF.EmitStmt(cast<CapturedStmt>(D.getAssociatedStmt())->getCapturedStmt());
  };

  emitInlinedDirective(CGF, OMPD_teams, CodeGen);
}
