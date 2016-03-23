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
#include "CodeGenFunction.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/IR/CallSite.h"

namespace clang {
namespace CodeGen {

class CGOpenMPRuntimeNVPTX : public CGOpenMPRuntime {
  //
  // Data Sharing related calls.
  //

  // \brief Return the address where the parallelism level is kept in shared
  // memory for the current thread. It is assumed we have up to 992 parallel
  // worker threads.
  //
  // FIXME: Make this value reside in a descriptor whose size is decided at
  // runtime (extern shared memory). This can be used for the other thread
  // specific state as well.
  LValue getParallelismLevelLValue(CodeGenFunction &CGF) const;

  // \brief Return an integer with the parallelism level. Zero means that the
  // current region is not enclosed in a parallel/simd region. The current level
  // is kept in a shared memory array.
  llvm::Value *getParallelismLevel(CodeGenFunction &CGF) const;

  // \brief Increase the value of parallelism level for the current thread.
  void increaseParallelismLevel(CodeGenFunction &CGF) const;

  // \brief Decrease the value of parallelism level for the current thread.
  void decreaseParallelismLevel(CodeGenFunction &CGF) const;

  // \brief Initialize with zero the value of parallelism level for the current
  // thread.
  void initializeParallelismLevel(CodeGenFunction &CGF) const;

  // \brief Type of the data sharing master slot. By default the size is zero meaning that the data size is to be determined.
  QualType DataSharingMasterSlotQtyFixedSize;
  QualType DataSharingMasterSlotQtyIncomplete;
  QualType getDataSharingMasterSlotQty(bool UseFixedDataSize = false);

  // \brief Type of the data sharing worker warp slot. By default the size is zero meaning that the data size is to be determined.
  QualType DataSharingWorkerWarpSlotQtyFixedSize;
  QualType DataSharingWorkerWarpSlotQtyIncomplete;
  QualType getDataSharingWorkerWarpSlotQty(bool UseFixedDataSize = false);

  // \brief Type of the data sharing root slot.
  QualType DataSharingRootSlotQty;
  QualType getDataSharingRootSlotQty();

  // \brief Return address of the initial slot that is used to share data.
  LValue getSharedDataRootSlotLValue(CodeGenFunction &CGF, bool IsMaster);

  // \brief Return the address where the address of the current slot is stored.
  LValue getSharedDataSlotPointerAddrLValue(CodeGenFunction &CGF, bool IsMaster);

  // \brief Return the address of the current data sharing slot.
  LValue getSharedDataSlotPointerLValue(CodeGenFunction &CGF, bool IsMaster);

  // \brief Return the address where the address of the current stack pointer (in the current slot) is stored.
  LValue getSharedDataStackPointerAddrLValue(CodeGenFunction &CGF, bool IsMaster);

  // \brief Return the address of the current data stack pointer.
  LValue getSharedDataStackPointerLValue(CodeGenFunction &CGF, bool IsMaster);

  // \brief Initialize the data sharing slots and pointers.
  void initializeSharedData(CodeGenFunction &CGF, bool IsMaster);

  //
  // NVPTX calls.
  //

  /// \brief Get the GPU warp size.
  llvm::Value *getNVPTXWarpSize(CodeGenFunction &CGF) const;

  /// \brief Get the id of the current thread on the GPU.
  llvm::Value *getNVPTXThreadID(CodeGenFunction &CGF) const;

  /// \brief Get the id of the current block on the GPU.
  llvm::Value *getNVPTXBlockID(CodeGenFunction &CGF) const;

  /// \brief Get the id of the warp in the block.
  llvm::Value *getNVPTXWarpID(CodeGenFunction &CGF) const;

  // \brief Get the maximum number of threads in a block of the GPU.
  llvm::Value *getNVPTXNumThreads(CodeGenFunction &CGF) const;

  // \brief Get a 32 bit mask, whose bits set to 1 represent the active threads.
  llvm::Value *getNVPTXWarpActiveThreadsMask(CodeGenFunction &CGF);

  // \brief Get the number of active threads in a warp.
  llvm::Value *getNVPTXWarpActiveNumThreads(CodeGenFunction &CGF);

  // \brief Get the ID of the thread among the current active threads in the warp.
  llvm::Value *getNVPTXWarpActiveThreadID(CodeGenFunction &CGF);

  // \brief Get a conditional that is set to true if the thread is the master of the active threads in the warp.
  llvm::Value *getNVPTXIsWarpActiveMaster(CodeGenFunction &CGF);

  /// \brief Get barrier to synchronize all threads in a block.
  void getNVPTXCTABarrier(CodeGenFunction &CGF) const;

  /// \brief Get barrier #n to synchronize selected (multiple of 32) threads in
  /// a block.
  void getNVPTXBarrier(CodeGenFunction &CGF, int ID, int NumThreads) const;

  // \brief Synchronize all GPU threads in a block.
  void syncCTAThreads(CodeGenFunction &CGF) const;

  //
  // OMP calls.
  //

  /// \brief Get the thread id of the OMP master thread.
  /// The master thread id is the first thread (lane) of the last warp in the
  /// GPU block.  Warp size is assumed to be some power of 2.
  /// Thread id is 0 indexed.
  /// E.g: If NumThreads is 33, master id is 32.
  ///      If NumThreads is 64, master id is 32.
  ///      If NumThreads is 1024, master id is 992.
  llvm::Value *getMasterThreadID(CodeGenFunction &CGF);

  /// \brief Get number of OMP workers for parallel region after subtracting
  /// the master warp.
  llvm::Value *getNumWorkers(CodeGenFunction &CGF);

  /// \brief Get thread id in team.
  /// FIXME: Remove the expensive remainder operation.
  llvm::Value *getTeamThreadId(CodeGenFunction &CGF);

  /// \brief Get global thread id.
  llvm::Value *getGlobalThreadId(CodeGenFunction &CGF);

  //
  // Private state and methods.
  //

  // Pointers to outlined function work for workers.
  llvm::SmallVector<llvm::Function *, 16> Work;

  class EntryFunctionState {
  public:
    llvm::BasicBlock *ExitBB;

    EntryFunctionState() : ExitBB(nullptr){};
  };

  class WorkerFunctionState {
  public:
    llvm::Function *WorkerFn;
    const CGFunctionInfo *CGFI;

    WorkerFunctionState(CodeGenModule &CGM);

  private:
    void createWorkerFunction(CodeGenModule &CGM);
  };

  // State information to track orphaned directives.
  bool IsOrphaned;
  // Track parallel nesting level.
  int ParallelNestingLevel;

  /// \brief Emit the worker function for the current target region.
  void emitWorkerFunction(WorkerFunctionState &WST);

  /// \brief Helper for worker function. Emit body of worker loop.
  void emitWorkerLoop(CodeGenFunction &CGF, WorkerFunctionState &WST);

  /// \brief Helper for target entry function. Guide the master and worker
  /// threads to their respective locations.
  void emitEntryHeader(CodeGenFunction &CGF, EntryFunctionState &EST,
                       WorkerFunctionState &WST);

  /// \brief Signal termination of OMP execution.
  void emitEntryFooter(CodeGenFunction &CGF, EntryFunctionState &EST);

  /// \brief Returns specified OpenMP runtime function for the current OpenMP
  /// implementation.  Specialized for the NVPTX device.
  /// \param Function OpenMP runtime function.
  /// \return Specified function.
  llvm::Constant *createNVPTXRuntimeFunction(unsigned Function);

  /// \brief Gets thread id value for the current thread.
  ///
  llvm::Value *getThreadID(CodeGenFunction &CGF, SourceLocation Loc) override;

  /// \brief Emits captured variables for the outlined function for the
  /// specified OpenMP parallel directive \a D.
  void
  emitCapturedVars(CodeGenFunction &CGF, const OMPExecutableDirective &S,
                   llvm::SmallVector<llvm::Value *, 16> &CapturedVars) override;

  //
  // Base class overrides.
  //

  /// \brief Creates offloading entry for the provided entry ID \a ID,
  /// address \a Addr and size \a Size.
  void createOffloadEntry(llvm::Constant *ID, llvm::Constant *Addr,
                          uint64_t Size) override;

  /// \brief Emit outlined function for 'target' directive on the NVPTX
  /// device.
  /// \param D Directive to emit.
  /// \param ParentName Name of the function that encloses the target region.
  /// \param OutlinedFn Outlined function value to be defined by this call.
  /// \param OutlinedFnID Outlined function ID value to be defined by this call.
  /// \param IsOffloadEntry True if the outlined function is an offload entry.
  /// An outlined function may not be an entry if, e.g. the if clause always
  /// evaluates to false.
  void emitTargetOutlinedFunction(const OMPExecutableDirective &D,
                                  StringRef ParentName,
                                  llvm::Function *&OutlinedFn,
                                  llvm::Constant *&OutlinedFnID,
                                  bool IsOffloadEntry) override;

  /// \brief Emit the code that each thread requires to execute when it
  /// encounters one of the three possible parallelism level. This also emits
  /// the required data sharing code for each level.
  /// \param Level0 Code to emit by the master thread when it encounters a
  /// parallel region.
  /// \param Level1 Code to emit by a worker thread when it encounters a
  /// parallel region.
  /// \param Sequential Code to emit by a worker thread when the parallel region
  /// is to be computed sequentially.
  void emitParallelismLevelCode(CodeGenFunction &CGF,
                                const RegionCodeGenTy &Level0,
                                const RegionCodeGenTy &Level1,
                                const RegionCodeGenTy &Sequential);

  //  // \brief Initialize state on entry to a target region.
  //  void enterTarget();
  //
  //  // \brief Reset state on exit from a target region.
  //  void exitTarget();

  // \brief Test if a construct is always encountered at nesting level 0.
  bool InL0();

  // \brief Test if a construct is always encountered at nesting level 1.
  bool InL1();

  // \brief Test if a construct is always encountered at nesting level 1 or
  // higher.
  bool InL1Plus();

  // \brief Test if the nesting level at which a construct is encountered is
  // indeterminate.  This happens for orphaned parallel directives.
  bool IndeterminateLevel();

public:
  explicit CGOpenMPRuntimeNVPTX(CodeGenModule &CGM);

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
  emitParallelOrTeamsOutlinedFunction(const OMPExecutableDirective &D,
                               const VarDecl *ThreadIDVar,
                               OpenMPDirectiveKind InnermostKind,
                               const RegionCodeGenTy &CodeGen) override;

  /// \brief Check if we should generate code as if \a ScheduleKind is static
  /// with a chunk size of 1.
  /// \param ScheduleKind Schedule Kind specified in the 'schedule' clause.
  /// \param Chunk size.
  ///
  bool generateCoalescedSchedule(OpenMPScheduleClauseKind ScheduleKind,
                                 bool ChunkSizeOne,
                                 bool ordered) const override;

  /// \brief Check if we must always generate a barrier at the end of a
  /// particular construct regardless of the presence of a nowait clause.
  /// This may occur when a particular offload device does not support
  /// concurrent execution of certain directive and clause combinations.
  bool requiresBarrier(const OMPLoopDirective &S) const override;
};

} // CodeGen namespace.
} // clang namespace.

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
