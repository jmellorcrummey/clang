///
/// Perform several offloading codegen tests
///

///##############################################
///
/// Test CodeGen for scalar reduction with parallel for
///
///##############################################

#ifdef TT1
// RUN:   %clang -fopenmp=libomp -target powerpc64le-ibm-linux-gnu -omptargets=nvptx64sm_35-nvidia-cuda \
// RUN:  -DTT1  -O0 -S -emit-llvm %s 2>&1
// RUN:   FileCheck -check-prefix=CK1 -input-file=parallel_for_reduction_codegen.ll.tgt-nvptx64sm_35-nvidia-cuda %s


void T1(){
  int a[1024], sum;

  for (int i = 0 ; i < 1024 ; i++) {
    a[i] = i;
  }

  sum = 0;

//CK1: call void @omp_reduction_op[[OPN:[0-9]*]](i8* %[[REF:[a-zA-a0-9\*()\"]+]], i8* %[[REF:[a-zA-a0-9\*()\"]+]], i8* %[[SIZEREF:[a-zA-a0-9\*()\"]+]])
//CK1: define internal void @omp_reduction_op(i8*, i8*, i8*)
//CK1-DAG: [[RES2:%[0-9]*]] = call i32 @__gpu_warpBlockRedu_fixed4_add(i32 [[PARA1:%[0-9]*]])

#pragma omp target  map(to:a[0:1024]) map(tofrom:sum) 
{
#pragma omp parallel for reduction(+:sum)
        for(int i=0; i<1024; i++) {
                sum += a[i];
        }
}//end target data
}

#endif

#ifdef TT2 
// RUN:   %clang -fopenmp=libomp -target powerpc64le-ibm-linux-gnu -omptargets=nvptx64sm_35-nvidia-cuda \
// RUN:  -DTT2  -O0 -S -emit-llvm %s 2>&1
// RUN:   FileCheck -check-prefix=CK2 -input-file=parallel_for_reduction_codegen.ll.tgt-nvptx64sm_35-nvidia-cuda %s
  
void T2(){
  double a[1024], sum;

  for (int i = 0 ; i < 1024 ; i++) {
    a[i] = i;
  }

  sum = 0;

//CK2-DAG: call void @omp_reduction_op[[OPN:[0-9]*]](i8* %[[REF:[a-zA-a0-9\*()\"]+]], i8* %[[REF:[a-zA-a0-9\*()\"]+]], i8* %[[SIZEREF:[a-zA-a0-9\*()\"]+]])
//CK2: define internal void @omp_reduction_op(i8*, i8*, i8*)
//CK2-DAG: [[RES2:%[0-9]*]] = call double @__gpu_warpBlockRedu_float8_mul(double [[PARA1:%[0-9]*]])

#pragma omp target  map(to:a[0:1024]) map(tofrom:sum) 
{
#pragma omp parallel for reduction(*:sum)
        for(int i=0; i<1024; i++) {
                sum *= a[i];
        }
}//end target data
}
#endif

  
