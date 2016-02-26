///
/// Perform a check of the combined loop code generation tests
/// (code within target region for nvptx)
///

///##############################################
///
/// Empty combined loop region (combined loop skeleton)
///
///##############################################

#ifdef TT1
// RUN:   %clang -fopenmp=libomp -target powerpc64le-ibm-linux-gnu -omptargets=nvptx64sm_35-nvidia-linux \
// RUN:   -DTT1 -O0 -S -emit-llvm %s 2>&1
// RUN:   FileCheck -check-prefix=CK1 -input-file=target_combined_loop_codegen_for_c.ll.tgt-nvptx64sm_35-nvidia-linux %s

// CK1: @__omptgt__[[KERNUNQ:[a-zA-Z0-9_\.]+]]__thread_limit = global i32 0
// CK1: @__omptgt__[[KERNUNQ]]__simd_info = constant i8 1

int foo() {

#pragma omp target teams distribute parallel for schedule(static, 1)
for(int i=0; i<100; i++){
}

  return 0;
}

// CK1: %[[OMPHANDLEADDR:[a-zA-Z0-9_\.]+]] = alloca i32
// CK1: store i32 %[[OMPHANDLE:[a-zA-Z0-9_\.]+]], i32* %[[OMPHANDLEADDR]]

// CK1: %[[TID0:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK1-NEXT: %[[TID1:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK1-NEXT: %[[TID2:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
// CK1-NEXT: %[[PROD:[a-zA-Z0-9_\.]+]] = mul i32 %[[TID1]], %[[TID2]]
// CK1-NEXT: %[[SUM:[a-zA-Z0-9_\.]+]] = add i32 %[[TID0]], %[[PROD]]
// CK1-NEXT: store i32 0, i32* %[[I:[a-zA-Z0-9_\.]+]]
// CK1-NEXT: %[[ORIGIDX1:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[I]], align 4
// CK1-NEXT: %[[CMP1:[a-zA-Z0-9_\.]+]] = icmp slt i32 %[[ORIGIDX1]], 100
// CK1-NEXT: br i1 %[[CMP1]], label %[[PRECOND:[a-zA-Z0-9_\.]+]], label %[[ENDCOMBFOR:[a-zA-Z0-9_\.]+]]

// CK1: [[PRECOND]]:
// CK1-NEXT: store i32 %[[SUM]], i32* %[[IDX:[a-zA-Z0-9_\.]+]]
// CK1-NEXT: br label %[[CONDCOMBFOR:[a-zA-Z0-9_\.]+]]

// CK1: [[CONDCOMBFOR]]:
// CK1-NEXT: %[[ITVAR:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[IDX]]
// CK1-NEXT: %[[COND:[a-zA-Z0-9_\.]+]] = icmp sle i32 %[[ITVAR]], 99
// CK1-NEXT: br i1 %[[COND]], label %[[BODYCOMBFOR:[a-zA-Z0-9_\.]+]], label %[[ENDCOMBFOR]]

// CK1: [[BODYCOMBFOR]]:
// CK1-NEXT: store i32 0, i32* %[[I]]
// CK1-NEXT: %[[REG7:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[IDX]]
// CK1-NEXT: %[[MUL:[a-zA-Z0-9_\.]+]] = mul nsw i32 %[[REG7]], 1
// CK1-NEXT: %[[REG8:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[I]]
// CK1-NEXT: %[[ADD1:[a-zA-Z0-9_\.]+]] = add nsw i32 %[[REG8]], %[[MUL]]
// CK1-NEXT: store i32 %[[ADD1]], i32* %[[I]]
// CK1-NEXT: br label %[[INCCOMBFOR:[a-zA-Z0-9_\.]+]]

// CK1: [[INCCOMBFOR]]:
// CK1-NEXT: %[[NCTAID:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
// CK1-NEXT: %[[NTID:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK1-NEXT: %[[REG11:[a-zA-Z0-9_\.]+]] = mul i32 %[[NCTAID]], %[[NTID]]
// CK1-NEXT: %[[REG12:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[IDX]]
// CK1-NEXT: %[[REG13:[a-zA-Z0-9_\.]+]] = add i32 %[[REG12]], %[[REG11]]
// CK1-NEXT: store i32 %[[REG13]], i32* %[[IDX]]
// CK1-NEXT: br label %[[CONDCOMBFOR]]

// CK1: [[ENDCOMBFOR]]:
// CK1-NEXT: ret void

#endif

#ifdef TT2
// RUN:   %clang -fopenmp=libomp -target powerpc64le-ibm-linux-gnu -omptargets=nvptx64sm_35-nvidia-linux \
// RUN:   -DTT2 -O0 -S -emit-llvm %s 2>&1
// RUN:   FileCheck -check-prefix=CK2 -input-file=target_combined_loop_codegen_for_c.ll.tgt-nvptx64sm_35-nvidia-linux %s

// CK2: @__omptgt__[[KERNUNQ:[a-zA-Z0-9_\.]+]]__thread_limit = global i32 0
// CK2: @__omptgt__[[KERNUNQ]]__simd_info = constant i8 1

int foo() {

  int red = 0;
  #pragma omp target teams distribute parallel for schedule(static, 1) reduction(+: red)
  for(int i=0; i<100; i++){
        red += i;
  }

  return 0;
}

// CK2: %[[ADDR:[a-zA-Z0-9_\.]+]] = alloca i32*
// CK2: %[[OMPHANDLEADDR:[a-zA-Z0-9_\.]+]] = alloca i32
// CK2: %[[REDRECVAR:[a-zA-Z0-9_\.]+]] = alloca { i32* }
// CK2: %[[REDRECSIZ:[a-zA-Z0-9_\.]+]] = alloca { i32* }
// CK2: %[[RED:[a-zA-Z0-9_\.]+]] = alloca i32
// CK2: %[[IDX:[a-zA-Z0-9_\.]+]] = alloca i32
// CK2: %[[TMP:[a-zA-Z0-9_\.]+]] = alloca { i32, i32, i32, i32, i8* }
// CK2: %[[REDADDRLHS:[a-zA-Z0-9_\.]+]] = alloca i32*
// CK2: %[[TMP4:[a-zA-Z0-9_\.]+]] = alloca { i32, i32, i32, i32, i8* }
// CK2: %[[TMP10:[a-zA-Z0-9_\.]+]] = alloca { i32, i32, i32, i32, i8* }
// CK2: %[[TMP15:[a-zA-Z0-9_\.]+]] = alloca { i32, i32, i32, i32, i8* }
// CK2: store i32* %0, i32** %[[ADDR]]
// CK2: store i32 %[[OMPHANDLE:[a-zA-Z0-9_\.]+]], i32* %[[OMPHANDLEADDR]]
// CK2: store i32 0, i32* %[[RED]]
// CK2: %[[REDADDR:[a-zA-Z0-9_\.]+]] = getelementptr { i32* }, { i32* }* %[[REDRECVAR]], i32 0, i32 0
// CK2: store i32* %[[RED]], i32** %[[REDADDR]]
// CK2-NEXT: br label %[[STARTCOMBFOR:[a-zA-Z0-9_\.]+]]

// CK2: [[STARTCOMBFOR]]:
// CK2: %[[TID0:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK2-NEXT: %[[TID1:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK2-NEXT: %[[TID2:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
// CK2-NEXT: %[[PROD:[a-zA-Z0-9_\.]+]] = mul i32 %[[TID1]], %[[TID2]]
// CK2-NEXT: %[[SUM:[a-zA-Z0-9_\.]+]] = add i32 %[[TID0]], %[[PROD]]
// CK2-NEXT: store i32 0, i32* %[[I:[a-zA-Z0-9_\.]+]]
// CK2-NEXT: %[[ORIGIDX1:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[I]], align 4
// CK2-NEXT: %[[CMP1:[a-zA-Z0-9_\.]+]] = icmp slt i32 %[[ORIGIDX1]], 100
// CK2-NEXT: br i1 %[[CMP1]], label %[[PRECOND:[a-zA-Z0-9_\.]+]], label %[[ENDCOMBFOR:[a-zA-Z0-9_\.]+]]

// CK2: [[PRECOND]]:
// CK2-NEXT: store i32 %[[SUM]], i32* %[[IDX]]
// CK2-NEXT: br label %[[CONDCOMBFOR:[a-zA-Z0-9_\.]+]]

// CK2: [[CONDCOMBFOR]]:
// CK2-NEXT: %[[ITVAR:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[IDX]]
// CK2-NEXT: %[[COND:[a-zA-Z0-9_\.]+]] = icmp sle i32 %[[ITVAR]], 99
// CK2-NEXT: br i1 %[[COND]], label %[[BODYCOMBFOR:[a-zA-Z0-9_\.]+]], label %[[ENDCOMBFOR]]

// CK2: [[BODYCOMBFOR]]:
// CK2-NEXT: store i32 0, i32* %[[I]]
// CK2-NEXT: %[[REG8:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[IDX]]
// CK2-NEXT: %[[MUL:[a-zA-Z0-9_\.]+]] = mul nsw i32 %[[REG8]], 1
// CK2-NEXT: %[[REG9:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[I]]
// CK2-NEXT: %[[ADD1:[a-zA-Z0-9_\.]+]] = add nsw i32 %[[REG9]], %[[MUL]]
// CK2-NEXT: store i32 %[[ADD1]], i32* %[[I]]
// CK2-NEXT: %[[REG10:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[I]]
// CK2-NEXT: %[[REG11:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[RED]]
// CK2-NEXT: %[[ADD2:[a-zA-Z0-9_\.]+]] = add nsw i32 %[[REG11]], %[[REG10]]
// CK2-NEXT: store i32 %[[ADD2]], i32* %[[RED]]
// CK2-NEXT: br label %[[INCCOMBFOR:[a-zA-Z0-9_\.]+]]

// CK2: [[INCCOMBFOR]]:
// CK2-NEXT: %[[NCTAID:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
// CK2-NEXT: %[[NTID:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK2-NEXT: %[[REG14:[a-zA-Z0-9_\.]+]] = mul i32 %[[NCTAID]], %[[NTID]]
// CK2-NEXT: %[[REG15:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[IDX]]
// CK2-NEXT: %[[REG16:[a-zA-Z0-9_\.]+]] = add i32 %[[REG15]], %[[REG14]]
// CK2-NEXT: store i32 %[[REG16]], i32* %[[IDX]]
// CK2-NEXT: br label %[[CONDCOMBFOR]]

// CK2: [[ENDCOMBFOR]]:
// CK2: call void @llvm.nvvm.barrier0()
// CK2-NEXT: br i1 true, label %[[GPUTHEN:[a-zA-Z0-9_\.]+]], label %[[GPUEND:[a-zA-Z0-9_\.]+]]

// CK2: [[GPUTHEN]]:
// CK2-NEXT: %[[VOIDREC:[a-zA-Z0-9_\.\"\*\(\)]+]] = bitcast { i32* }* %[[REDRECVAR]] to i8*
// CK2-NEXT: %[[VOIDSIZ:[a-zA-Z0-9_\.\"\*\(\)]+]] = bitcast { i32* }* %[[REDRECSIZ]] to i8*
// CK2-NEXT: call void @omp_reduction_op(i8* %[[VOIDREC]], i8* %[[VOIDREC]], i8* %[[VOIDSIZ]])
// CK2-NEXT: br label %[[GPUEND]]

// CK2: [[GPUEND]]:
// CK2-NEXT: %[[TID00:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
// CK2-NEXT: %[[TID10:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK2-NEXT: %[[TID20:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK2-NEXT: %[[PROD0:[a-zA-Z0-9_\.]+]] = mul i32 %[[TID00]], %[[TID10]]
// CK2-NEXT: %[[GID:[a-zA-Z0-9_\.]+]] = add i32  %[[PROD0]], %[[TID20]]
// CK2-NEXT: %[[VOIDREC2:[a-zA-Z0-9_\.\"\*\(\)]+]] = bitcast { i32* }* %[[REDRECVAR]] to i8*
// CK2-NEXT: %[[VOIDSIZ2:[a-zA-Z0-9_\.\"\*\(\)]+]] = bitcast { i32* }* %[[REDRECSIZ]] to i8*
// CK2-NEXT: %[[REG18:[a-zA-Z0-9_\.]+]] = call i32 @__kmpc_reduce_combined({ i32, i32, i32, i32, i8* }* %[[TMP]])
// CK2-NEXT: switch i32 %[[REG18]], label %[[REDCONT:[a-zA-Z0-9_\.]+]]
// CK2-NEXT: i32 2, label %[[REDCASE2:[a-zA-Z0-9_\.]+]]

// CK2: [[REDCASE2]]:
// CK2-NEXT: %[[BID5:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
// CK2-NEXT: %[[BSIZE6:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK2-NEXT: %[[TID7:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK2-NEXT: %[[REG24:[a-zA-Z0-9_\.]+]] = mul i32 %[[BID5]], %[[BSIZE6]]
// CK2-NEXT: %[[GID8:[a-zA-Z0-9_\.]+]] = add i32 %[[REG24]], %[[TID7]]
// CK2-NEXT: %[[REDADDRRHS9:[a-zA-Z0-9_\.]+]] = getelementptr { i32* }, { i32* }* %[[REDRECVAR]], i32 0, i32 0
// CK2-NEXT: %[[REDRHS:[a-zA-Z0-9_\.]+]] = load i32*, i32** %[[REDADDRRHS9]]
// CK2-NEXT: %[[REG25:[a-zA-Z0-9_\.]+]] =  load i32, i32* %[[REDRHS]]
// CK2-NEXT: call void @__kmpc_atomic_fixed4_add({ i32, i32, i32, i32, i8* }* %[[TMP4]], i32 %[[GID8]], i32* %0, i32 %[[REG25]])
// CK2-NEXT: %[[BID16:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
// CK2-NEXT: %[[BSIZE17:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK2-NEXT: %[[TID18:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK2-NEXT: %[[REG26:[a-zA-Z0-9_\.]+]] = mul i32 %[[BID16]], %[[BSIZE17]]
// CK2-NEXT: %[[GID19:[a-zA-Z0-9_\.]+]] = add i32 %[[REG26]], %[[TID18]]
// CK2-NEXT: call void @__kmpc_end_reduce({ i32, i32, i32, i32, i8* }* %[[TMP15]], i32 %[[GID19]], [8 x i32]* @.lck.)
// CK2-NEXT: br label %[[REDCONT]]

// CK2: [[REDCONT]]:
// CK2-NEXT: br label %[[SAFETYNET:[a-zA-Z0-9_\.]+]]

// CK2: [[SAFETYNET]]:
// CK2-NEXT: ret void

#endif
