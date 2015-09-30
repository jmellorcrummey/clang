///
/// Perform several driver tests for OpenMP offloading
///

/// ###########################################################################

/// Check whether an invalid OpenMP target is specified:
// RUN:   %clang -### -fopenmp=libomp -omptargets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// CHK-INVALID-TARGET: error: OpenMP target is invalid: 'aaa-bbb-ccc-ddd'

/// ###########################################################################

/// Check error for empty -omptargets
// RUN:   %clang -### -fopenmp=libiomp5 -omptargets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-OMPTARGETS %s
// CHK-EMPTY-OMPTARGETS: warning: joined argument expects additional value: '-omptargets='

/// ###########################################################################

/// Check whether we are using a target whose toolchain was not prepared to
/// to support offloading - e.g. x86_64-apple-darwin:
// RUN:   %clang -### -fopenmp=libiomp5 -omptargets=x86_64-apple-darwin %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-SUPPORT %s
// CHK-NO-SUPPORT: error: Toolchain for target 'x86_64-apple-darwin' is not supporting OpenMP offloading.

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// Each target phase must be binded to a target and linked into a shared
/// library. The host compiler phase result is used in the compiler phase of the
/// the target
// RUN:   %clang -ccc-print-phases -fopenmp=libiomp5 -target powerpc64-ibm-linux-gnu -omptargets=x86_64-pc-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s

// Host linking
// CHK-PHASES-DAG: {{.*}}: linker, {[[A0:[0-9]+]], [[BL1:[0-9]+]]}, image

// Target 1 library generation
// CHK-PHASES-DAG: [[BL1]]: bind-target, {[[L1:[0-9]+]]}, shared-object
// CHK-PHASES-DAG: [[L1]]: linker, {[[BA1:[0-9]+]]}, shared-object
// CHK-PHASES-DAG: [[BA1]]: bind-target, {[[A1:[0-9]+]]}, object
// CHK-PHASES-DAG: [[A1]]: assembler, {[[BB1:[0-9]+]]}, object
// CHK-PHASES-DAG: [[BB1]]: bind-target, {[[B1:[0-9]+]]}, assembler
// CHK-PHASES-DAG: [[B1]]: backend, {[[BC1:[0-9]+]]}, assembler
// CHK-PHASES-DAG: [[BC1]]: bind-target, {[[C1:[0-9]+]]}, ir
// CHK-PHASES-DAG: [[C1]]: compiler, {[[BP1:[0-9]+]], [[BC01:[0-9]+]]}, ir
// CHK-PHASES-DAG: [[BC01]]: bind-target, {[[C0:[0-9]+]]}, ir
// CHK-PHASES-DAG: [[BP1]]: bind-target, {[[P1:[0-9]+]]}, cpp-output
// CHK-PHASES-DAG: [[P1]]: preprocessor, {[[I:[0-9]+]]}, cpp-output

// Host objects generation:
// CHK-PHASES-DAG: [[A0]]: assembler, {[[B0:[0-9]+]]}, object
// CHK-PHASES-DAG: [[B0]]: backend, {[[C0]]}, assembler
// CHK-PHASES-DAG: [[C0]]: compiler, {[[P0:[0-9]+]]}, ir
// CHK-PHASES-DAG: [[P0]]: preprocessor, {[[I]]}, cpp-output

// Single input file:
// CHK-PHASES-DAG: [[I]]: input, {{.*}}, c

/// ###########################################################################

/// Check the phases graph when using two targets, and one of them is the same
/// as the host. We also add a library to make sure they are not treated has
/// inputs.
/// Each target phase must be binded to a target and linked into a shared
/// library. The host compiler phase result is used in the compiler phase of the
/// the target
// RUN:   %clang -ccc-print-phases -lm -fopenmp=libiomp5 -target powerpc64-ibm-linux-gnu -omptargets=x86_64-pc-linux-gnu,powerpc64-ibm-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES2 %s

// Host linking
// CHK-PHASES2-DAG: {{.*}}: linker, {[[L:[0-9]+]], [[A0:[0-9]+]], [[BL1:[0-9]+]], [[BL2:[0-9]+]]}, image

// Target 2 library generation
// CHK-PHASES2-DAG: [[BL2]]: bind-target, {[[L2:[0-9]+]]}, shared-object
// CHK-PHASES2-DAG: [[L2]]: linker, {[[L]], [[BA2:[0-9]+]]}, shared-object
// CHK-PHASES2-DAG: [[BA2]]: bind-target, {[[A2:[0-9]+]]}, object
// CHK-PHASES2-DAG: [[A2]]: assembler, {[[BB2:[0-9]+]]}, object
// CHK-PHASES2-DAG: [[BB2]]: bind-target, {[[B2:[0-9]+]]}, assembler
// CHK-PHASES2-DAG: [[B2]]: backend, {[[BC2:[0-9]+]]}, assembler
// CHK-PHASES2-DAG: [[BC2]]: bind-target, {[[C2:[0-9]+]]}, ir
// CHK-PHASES2-DAG: [[C2]]: compiler, {[[BP2:[0-9]+]], [[BC02:[0-9]+]]}, ir
// CHK-PHASES2-DAG: [[BC02]]: bind-target, {[[C0:[0-9]+]]}, ir
// CHK-PHASES2-DAG: [[BP2]]: bind-target, {[[P2:[0-9]+]]}, cpp-output
// CHK-PHASES2-DAG: [[P2]]: preprocessor, {[[I:[0-9]+]]}, cpp-output

// Target 1 library generation
// CHK-PHASES2-DAG: [[BL1]]: bind-target, {[[L1:[0-9]+]]}, shared-object
// CHK-PHASES2-DAG: [[L1]]: linker, {[[L]], [[BA1:[0-9]+]]}, shared-object
// CHK-PHASES2-DAG: [[BA1]]: bind-target, {[[A1:[0-9]+]]}, object
// CHK-PHASES2-DAG: [[A1]]: assembler, {[[BB1:[0-9]+]]}, object
// CHK-PHASES2-DAG: [[BB1]]: bind-target, {[[B1:[0-9]+]]}, assembler
// CHK-PHASES2-DAG: [[B1]]: backend, {[[BC1:[0-9]+]]}, assembler
// CHK-PHASES2-DAG: [[BC1]]: bind-target, {[[C1:[0-9]+]]}, ir
// CHK-PHASES2-DAG: [[C1]]: compiler, {[[BP1:[0-9]+]], [[BC01:[0-9]+]]}, ir
// CHK-PHASES2-DAG: [[BC01]]: bind-target, {[[C0]]}, ir
// CHK-PHASES2-DAG: [[BP1]]: bind-target, {[[P1:[0-9]+]]}, cpp-output
// CHK-PHASES2-DAG: [[P1]]: preprocessor, {[[I:[0-9]+]]}, cpp-output

// Host objects generation:
// CHK-PHASES2-DAG: [[A0]]: assembler, {[[B0:[0-9]+]]}, object
// CHK-PHASES2-DAG: [[B0]]: backend, {[[C0]]}, assembler
// CHK-PHASES2-DAG: [[C0]]: compiler, {[[P0:[0-9]+]]}, ir
// CHK-PHASES2-DAG: [[P0]]: preprocessor, {[[I]]}, cpp-output

// Single input file:
// CHK-PHASES2-DAG: [[I]]: input, {{.*}}, c
// CHK-PHASES2-DAG: [[L]]: input, "m", object

/// ###########################################################################

/// Check of the commands passed to each tool when using valid OpenMP targets.
/// Here we also check that offloading does not break the use of integrated
/// assembler. It does however preclude the use of integrated preprocessor as
/// host IR is shared by all the compile phases. There several offloading
/// specific commands:
/// -omp-target-mode: will tell the frontend that it will generate code for a
/// target.
/// -omp-main-file-path: the original source file that relates with that
/// frontend run, will be used to generate unique variable names (IDs) that are
/// the same for all targets.
/// -omp-host-output-file-path: specifies the host IR file that can be loaded by
/// the target code generation to gather information about which declaration
/// really need to be emitted.
///
// RUN:   %clang -### -fopenmp=libiomp5 -target powerpc64le-linux -omptargets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-COMMANDS %s
//

// Final linking - host (ppc64le)
// CHK-COMMANDS-DAG: ld" {{.*}}"-m" "elf64lppc" {{.*}}"-o" "a.out" {{.*}}"[[HSTOBJ:.+]].o" "-liomp5" "-lomptarget" {{.*}}"-T" "[[LKSCRIPT:.+]].lk"

// Target 2 commands (x86_64)
// CHK-COMMANDS-DAG: ld" {{.*}}"-m" "elf_x86_64" {{.*}}"-shared" {{.*}}"-o" "[[T2LIB:.+]].so" {{.*}}"[[T2OBJ:.+]].o" {{.*}}"-liomp5"
// CHK-COMMANDS-DAG: clang{{.*}}" "-cc1" {{.*}}"-fopenmp=libiomp5" "-omptargets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu" "-omp-target-mode" "-omp-main-file-path" "[[SRC:.+]].c" {{.*}}"-triple" "x86_64-pc-linux-gnu" {{.*}}"-emit-obj" {{.*}}"-o" "[[T2OBJ]].o" "-x" "ir" "[[T2BC:.+]].bc"
// CHK-COMMANDS-DAG: clang{{.*}}" "-cc1" {{.*}}"-fopenmp=libiomp5" "-omptargets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu" "-omp-target-mode" "-omp-host-output-file-path" "[[HSTBC:.+]].bc" "-omp-main-file-path" "[[SRC]].c" {{.*}}"-triple" "x86_64-pc-linux-gnu" {{.*}}"-emit-llvm-bc" {{.*}}"-o" "[[T2BC]].bc" "-x" "cpp-output" "[[T2PP:.+]].i"
// CHK-COMMANDS-DAG: clang{{.*}}" "-cc1" {{.*}}"-fopenmp=libiomp5" "-omptargets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu" "-omp-target-mode" "-omp-main-file-path" "[[SRC]].c" {{.*}}"-triple" "x86_64-pc-linux-gnu" {{.*}}"-E" {{.*}}"-o" "[[T2PP]].i" "-x" "c" "[[SRC]].c"

// Target 1 commands (ppc64le)
// CHK-COMMANDS-DAG: ld" {{.*}}"-m" "elf64lppc" {{.*}}"-shared" {{.*}}"-o" "[[T1LIB:.+]].so" {{.*}}"[[T1OBJ:.+]].o" {{.*}}"-liomp5"
// CHK-COMMANDS-DAG: clang{{.*}}" "-cc1" {{.*}}"-fopenmp=libiomp5" "-omptargets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu" "-omp-target-mode" "-omp-main-file-path" "[[SRC]].c" {{.*}}"-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-emit-obj" {{.*}}"-o" "[[T1OBJ]].o" "-x" "ir" "[[T1BC:.+]].bc"
// CHK-COMMANDS-DAG: clang{{.*}}" "-cc1" {{.*}}"-fopenmp=libiomp5" "-omptargets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu" "-omp-target-mode" "-omp-host-output-file-path" "[[HSTBC]].bc" "-omp-main-file-path" "[[SRC]].c" {{.*}}"-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-emit-llvm-bc" {{.*}}"-o" "[[T1BC]].bc" "-x" "cpp-output" "[[T1PP:.+]].i"
// CHK-COMMANDS-DAG: clang{{.*}}" "-cc1" {{.*}}"-fopenmp=libiomp5" "-omptargets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu" "-omp-target-mode" "-omp-main-file-path" "[[SRC]].c" {{.*}}"-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-E" {{.*}}"-o" "[[T1PP]].i" "-x" "c" "[[SRC]].c"

// Host object generation
// CHK-COMMANDS-DAG: clang{{.*}}" "-cc1" {{.*}}"-fopenmp=libiomp5" "-omptargets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu" "-omp-main-file-path" "[[SRC]].c" {{.*}}"-triple" "powerpc64le--linux" {{.*}}"-emit-obj" {{.*}}"-o" "[[HSTOBJ]].o" "-x" "ir" "[[HSTBC]].bc"
// CHK-COMMANDS-DAG: clang{{.*}}" "-cc1" {{.*}}"-fopenmp=libiomp5" "-omptargets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu" "-omp-main-file-path" "[[SRC]].c" {{.*}}"-triple" "powerpc64le--linux" {{.*}}"-emit-llvm-bc" {{.*}}"-o" "[[HSTBC]].bc" "-x" "c" "[[SRC]].c"

/// ###########################################################################

/// Check the automatic detection of target files. The driver will automatically
/// detect if a target file is in the same path as the host file and include
/// that in the compilation. The user can choose to have the compiler generating
/// a warning if such file is included.
/// Create dummy host and target files.
// RUN:   echo ' ' > %t.i
// RUN:   echo ' ' > %t.i.tgt-x86_64-pc-linux-gnu
// RUN:   echo ' ' > %t.bc
// RUN:   echo ' ' > %t.bc.tgt-x86_64-pc-linux-gnu
// RUN:   echo ' ' > %t.s
// RUN:   echo ' ' > %t.s.tgt-x86_64-pc-linux-gnu
// RUN:   echo ' ' > %t.o
// RUN:   echo ' ' > %t.o.tgt-x86_64-pc-linux-gnu

// RUN:   %clang -### -fopenmp=libiomp5 -target powerpc64-linux -omptargets=x86_64-pc-linux-gnu %t.i -c -emit-llvm -Womp-implicit-target-files 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET-WARN-IMPLICIT %s
// RUN:   %clang -### -fopenmp=libiomp5 -target powerpc64-linux -omptargets=x86_64-pc-linux-gnu %t.i -c -emit-llvm 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET-NOTWARN-IMPLICIT %s
// RUN:   %clang -### -fopenmp=libiomp5 -target powerpc64-linux -omptargets=x86_64-pc-linux-gnu %t.bc -S -Womp-implicit-target-files 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET-WARN-IMPLICIT %s
// RUN:   %clang -### -fopenmp=libiomp5 -target powerpc64-linux -omptargets=x86_64-pc-linux-gnu %t.bc -S 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET-NOTWARN-IMPLICIT %s
// RUN:   %clang -### -fopenmp=libiomp5 -target powerpc64-linux -omptargets=x86_64-pc-linux-gnu %t.s -c -Womp-implicit-target-files 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET-WARN-IMPLICIT %s
// RUN:   %clang -### -fopenmp=libiomp5 -target powerpc64-linux -omptargets=x86_64-pc-linux-gnu %t.s -c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET-NOTWARN-IMPLICIT %s
// RUN:   %clang -### -fopenmp=libiomp5 -target powerpc64-linux -omptargets=x86_64-pc-linux-gnu %t.o -Womp-implicit-target-files 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET-WARN-IMPLICIT %s
// RUN:   %clang -### -fopenmp=libiomp5 -target powerpc64-linux -omptargets=x86_64-pc-linux-gnu %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET-NOTWARN-IMPLICIT %s

// CHK-TARGET-WARN-IMPLICIT: warning: OpenMP target file '{{.*}}.tgt-x86_64-pc-linux-gnu' is being implicitly used in the 'x86_64-pc-linux-gnu' toolchain.
// CHK-TARGET-NOTWARN-IMPLICIT-NOT: warning: OpenMP target file '{{.*}}.tgt-x86_64-pc-linux-gnu' is being implicitly used in the 'x86_64-pc-linux-gnu' toolchain.

/// ###########################################################################

/// Check separate compilation feature - the ability of the driver to assign
/// host and target files to different phases. Only the host files are passed
/// to the driver. The driver will detect all the target files
// RUN:   echo ' ' > %t.1.s
// RUN:   echo ' ' > %t.1.s.tgt-x86_64-pc-linux-gnu
// RUN:   echo ' ' > %t.2.o
// RUN:   %clang -### -fopenmp=libiomp5 -target powerpc64-linux -omptargets=x86_64-pc-linux-gnu %t.1.s %t.2.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SEP-COMPILATION %s

// Final linking
// CHK-SEP-COMPILATION-DAG: ld" {{.*}}"-m" "elf64ppc" {{.*}}"[[HOSTOBJ:.+]].o" "[[HOSTOBJ2:.+]].o" "-liomp5" "-lomptarget" {{.*}}"-T" "[[LKS:.+]].lk"

// Target image generation
// CHK-SEP-COMPILATION-DAG: ld" {{.*}}"-m" "elf_x86_64" {{.*}}"-shared" {{.*}}"-o" "[[TGTSO:.+]].so" {{.*}}"[[TGTOBJ:.+]].o" {{.*}}"-liomp5"
// CHK-SEP-COMPILATION-DAG: clang{{.*}}" "-cc1as" {{.*}}"-triple" "x86_64-pc-linux-gnu" {{.*}}"-o" "[[TGTOBJ]].o" "[[TGTASM:.+]].s.tgt-x86_64-pc-linux-gnu"

// Host object generation
// CHK-SEP-COMPILATION-DAG: clang{{.*}}" "-cc1as" {{.*}}"-triple" "powerpc64--linux" {{.*}}"-o" "[[HOSTOBJ]].o" "[[HOSTASM:.+]].s"
