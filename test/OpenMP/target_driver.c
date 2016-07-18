///
/// Perform several driver tests for OpenMP offloading
///

/// ###########################################################################

/// Check whether an invalid OpenMP target is specified:
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// CHK-INVALID-TARGET: error: OpenMP target is invalid: 'aaa-bbb-ccc-ddd'

/// ###########################################################################

/// Check warning for empty -fopenmp-targets
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-OMPTARGETS %s
// CHK-EMPTY-OMPTARGETS: warning: joined argument expects additional value: '-fopenmp-targets='

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// The actions should be exactly the same as if not offloading was being used.
// RUN:   %clang -ccc-print-phases -fopenmp=libomp -target powerpc64-ibm-linux-gnu -fopenmp-targets=x86_64-pc-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s

// CHK-PHASES-DAG: {{.*}}: linker, {[[A0:[0-9]+]]}, image
// CHK-PHASES-DAG: [[A0]]: assembler, {[[A1:[0-9]+]]}, object
// CHK-PHASES-DAG: [[A1]]: backend, {[[A2:[0-9]+]]}, assembler
// CHK-PHASES-DAG: [[A2]]: compiler, {[[A3:[0-9]+]]}, ir
// CHK-PHASES-DAG: [[A3]]: preprocessor, {[[I:[0-9]+]]}, cpp-output
// CHK-PHASES-DAG: [[I]]: input, {{.*}}, c

/// ###########################################################################

/// Check the phases when using multiple targets. Again, the actions are the
/// same as if no offloading was being used. Here we also add a library to make
/// sure it is not treated as input.
// RUN:   %clang -ccc-print-phases -lm -fopenmp=libomp -target powerpc64-ibm-linux-gnu -fopenmp-targets=x86_64-pc-linux-gnu,powerpc64-ibm-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-LIB %s

// CHK-PHASES-LIB-DAG: {{.*}}: linker, {[[L0:[0-9]+]], [[A0:[0-9]+]]}, image
// CHK-PHASES-LIB-DAG: [[A0]]: assembler, {[[A1:[0-9]+]]}, object
// CHK-PHASES-LIB-DAG: [[A1]]: backend, {[[A2:[0-9]+]]}, assembler
// CHK-PHASES-LIB-DAG: [[A2]]: compiler, {[[A3:[0-9]+]]}, ir
// CHK-PHASES-LIB-DAG: [[A3]]: preprocessor, {[[I:[0-9]+]]}, cpp-output
// CHK-PHASES-LIB-DAG: [[I]]: input, {{.*}}, c
// CHK-PHASES-LIB-DAG: [[L0]]: input, "m", object

/// ###########################################################################

/// Check the phases when using multiple targets and passing an object file as
/// input. An unbundling action has to be created.
// RUN:   echo 'bla' > %t.o
// RUN:   %clang -ccc-print-phases -lm -fopenmp=libomp -target powerpc64-ibm-linux-gnu -fopenmp-targets=x86_64-pc-linux-gnu,powerpc64-ibm-linux-gnu %s %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-OBJ %s

// CHK-PHASES-OBJ-DAG: {{.*}}: linker, {[[L0:[0-9]+]], [[A0:[0-9]+]], [[B0:[0-9]+]]}, image
// CHK-PHASES-OBJ-DAG: [[A0]]: assembler, {[[A1:[0-9]+]]}, object
// CHK-PHASES-OBJ-DAG: [[A1]]: backend, {[[A2:[0-9]+]]}, assembler
// CHK-PHASES-OBJ-DAG: [[A2]]: compiler, {[[A3:[0-9]+]]}, ir
// CHK-PHASES-OBJ-DAG: [[A3]]: preprocessor, {[[I:[0-9]+]]}, cpp-output
// CHK-PHASES-OBJ-DAG: [[I]]: input, {{.*}}, c
// CHK-PHASES-OBJ-DAG: [[L0]]: input, "m", object
// CHK-PHASES-OBJ-DAG: [[B0]]: clang-offload-unbundler, {[[B1:[0-9]+]]}, object
// CHK-PHASES-OBJ-DAG: [[B1]]: input, "{{.*}}.o", object

/// ###########################################################################

/// Check the phases when using multiple targets and separate compilation.
// RUN:   echo 'bla' > %t.s
// RUN:   %clang -ccc-print-phases -c -lm -fopenmp=libomp -target powerpc64-ibm-linux-gnu -fopenmp-targets=x86_64-pc-linux-gnu,powerpc64-ibm-linux-gnu %t.s -x cpp-output %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-SEP %s

// CHK-PHASES-SEP-DAG: [[A:[0-9]+]]: input, "{{.*}}.c", cpp-output
// CHK-PHASES-SEP-DAG: [[A1:[0-9]+]]: clang-offload-unbundler, {[[A]]}, cpp-output
// CHK-PHASES-SEP-DAG: [[A2:[0-9]+]]: compiler, {[[A1]]}, ir
// CHK-PHASES-SEP-DAG: [[A3:[0-9]+]]: backend, {[[A2]]}, assembler
// CHK-PHASES-SEP-DAG: [[A4:[0-9]+]]: assembler, {[[A3]]}, object
// CHK-PHASES-SEP-DAG: {{.*}}: clang-offload-bundler, {[[A4]]}, object

// CHK-PHASES-SEP-DAG: [[B:[0-9]+]]: input, "{{.*}}.s", assembler
// CHK-PHASES-SEP-DAG: [[B1:[0-9]+]]: clang-offload-unbundler, {[[B]]}, assembler
// CHK-PHASES-SEP-DAG: [[B2:[0-9]+]]: assembler, {[[B1]]}, object
// CHK-PHASES-SEP-DAG: {{.*}}: clang-offload-bundler, {[[B2]]}, object

/// ###########################################################################

/// Check of the commands passed to each tool when using valid OpenMP targets.
/// Here we also check that offloading does not break the use of integrated
/// assembler. It does however preclude the use of integrated preprocessor as
/// host IR is shared by all the compile phases. There are also two offloading
/// specific commands:
/// -fopenmp-is-device: will tell the frontend that it will generate code for a
/// target.
/// -fopenmp-host-ir-file-path: specifies the host IR file that can be loaded by
/// the target code generation to gather information about which declaration
/// really need to be emitted.
///
// RUN:   %clang -### -fopenmp=libomp -target powerpc64le-linux -fopenmp-targets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-COMMANDS %s
// RUN:   %clang -### -fopenmp=libomp -target powerpc64le-linux -fopenmp-targets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu %s -save-temps 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-COMMANDS-ST %s
//

// Final linking - host (ppc64le)
// CHK-COMMANDS-DAG:    ld" {{.*}}"-m" "elf64lppc" {{.*}}"-o" "a.out" {{.*}}"[[HSTOBJ:.+]].o" "-lomp" "-lomptarget" {{.*}}"-T" "[[LKSCRIPT:.+]].lk"
// CHK-COMMANDS-ST-DAG: ld" {{.*}}"-m" "elf64lppc" {{.*}}"-o" "a.out" {{.*}}"[[HSTOBJ:.+]].o" "-lomp" "-lomptarget" {{.*}}"-T" "[[LKSCRIPT:.+]].lk"

// Target 2 commands (x86_64)
// CHK-COMMANDS-DAG:    ld" {{.*}}"-m" "elf_x86_64" {{.*}}"-shared" {{.*}}"-o" "[[T2LIB:.+]]" {{.*}}"[[T2OBJ:.+]].o" {{.*}}"-lomp"
// CHK-COMMANDS-DAG:    clang{{.*}}" "-cc1" "-triple" "x86_64-pc-linux-gnu" {{.*}}"-emit-obj" {{.*}}"-fopenmp" {{.*}}"-o" "[[T2OBJ]].o" "-x" "ir" "[[T2BC:.+]].bc"
// CHK-COMMANDS-DAG:    clang{{.*}}" "-cc1" "-triple" "x86_64-pc-linux-gnu" {{.*}}"-emit-llvm-bc" {{.*}}"-fopenmp" {{.*}}"-o" "[[T2BC]].bc" "-x" "c" "[[SRC:.+]].c" "-fopenmp-is-device" "-fopenmp-host-ir-file-path" "[[HSTBC:.+]].bc"

// CHK-COMMANDS-ST-DAG:    ld" {{.*}}"-m" "elf_x86_64" {{.*}}"-shared" {{.*}}"-o" "[[T2LIB:.+]]" {{.*}}"[[T2OBJ:.+]].o" {{.*}}"-lomp"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1as" "-triple" "x86_64-pc-linux-gnu" "-filetype" "obj" {{.*}}"-o" "[[T2OBJ]].o" "[[T2ASM:.+]].s"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "x86_64-pc-linux-gnu" {{.*}}"-S" {{.*}}"-fopenmp" {{.*}}"-o" "[[T2ASM]].s" "-x" "ir" "[[T2BC:.+]].bc"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "x86_64-pc-linux-gnu" {{.*}}"-emit-llvm-bc" {{.*}}"-fopenmp" {{.*}}"-o" "[[T2BC]].bc" "-x" "cpp-output" "[[T2PP:.+]].i" "-fopenmp-is-device" "-fopenmp-host-ir-file-path" "[[HSTBC:.+]].bc"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "x86_64-pc-linux-gnu" {{.*}}"-E" {{.*}}"-fopenmp" {{.*}}"-o" "[[T2PP]].i" "-x" "c" "[[SRC:.+]].c"

// Target 1 commands (ppc64le)
// CHK-COMMANDS-DAG:    ld" {{.*}}"-m" "elf64lppc" {{.*}}"-shared" {{.*}}"-o" "[[T1LIB:.+]]" {{.*}}"[[T1OBJ:.+]].o" {{.*}}"-lomp"
// CHK-COMMANDS-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-emit-obj" {{.*}}"-fopenmp" {{.*}}"-o" "[[T1OBJ]].o" "-x" "ir" "[[T1BC:.+]].bc"
// CHK-COMMANDS-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-emit-llvm-bc" {{.*}}"-fopenmp" {{.*}}"-o" "[[T1BC]].bc" "-x" "c" "[[SRC]].c" "-fopenmp-is-device" "-fopenmp-host-ir-file-path" "[[HSTBC]].bc"

// CHK-COMMANDS-ST-DAG:    ld" {{.*}}"-m" "elf64lppc" {{.*}}"-shared" {{.*}}"-o" "[[T1LIB:.+]]" {{.*}}"[[T1OBJ:.+]].o" {{.*}}"-lomp"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1as" "-triple" "powerpc64le-ibm-linux-gnu" "-filetype" "obj" {{.*}}"-o" "[[T1OBJ]].o" "[[T1ASM:.+]].s"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-S" {{.*}}"-fopenmp" {{.*}}"-o" "[[T1ASM]].s" "-x" "ir" "[[T1BC:.+]].bc"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-emit-llvm-bc" {{.*}}"-fopenmp" {{.*}}"-o" "[[T1BC]].bc" "-x" "cpp-output" "[[T1PP:.+]].i" "-fopenmp-is-device" "-fopenmp-host-ir-file-path" "[[HSTBC]].bc"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-E" {{.*}}"-fopenmp" {{.*}}"-o" "[[T1PP]].i" "-x" "c" "[[SRC]].c"

// Host object generation
// CHK-COMMANDS-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le--linux" {{.*}}"-emit-obj" {{.*}}"-fopenmp" {{.*}}"-o" "[[HSTOBJ]].o" "-x" "ir" "[[HSTBC]].bc"
// CHK-COMMANDS-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le--linux" {{.*}}"-emit-llvm-bc"{{.*}}"-fopenmp" {{.*}}"-o" "[[HSTBC]].bc" "-x" "c" "[[SRC]].c" "-fopenmp-targets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu"

// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1as" "-triple" "powerpc64le--linux" "-filetype" "obj" {{.*}}"-o" "[[HSTOBJ]].o" "[[HSTASM:.+]].s"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le--linux" {{.*}}"-S"{{.*}}"-fopenmp" {{.*}}"-o" "[[HSTASM]].s" "-x" "ir" "[[HSTBC:.+]].bc"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le--linux" {{.*}}"-emit-llvm-bc"{{.*}}"-fopenmp" {{.*}}"-o" "[[HSTBC]].bc" "-x" "cpp-output" "[[HSTPP:.+]].i" "-fopenmp-targets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu"
// CHK-COMMANDS-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le--linux" {{.*}}"-E"{{.*}}"-fopenmp" {{.*}}"-o" "[[HSTPP]].i" "-x" "c" "[[SRC]].c"

/// ###########################################################################

/// Check separate compilation
///
// RUN:   echo 'bla' > %t.s
// RUN:   %clang -### -fopenmp=libomp -c -target powerpc64le-linux -fopenmp-targets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu %t.s -x cpp-output %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-COMMANDS-SEP %s
// RUN:   %clang -### -fopenmp=libomp -c -target powerpc64le-linux -fopenmp-targets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu %t.s -x cpp-output %s -save-temps 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-COMMANDS-SEP-ST %s
//

// Unbundle the input files.
// CHK-COMMANDS-SEP-DAG:    clang-offload-bundler{{.*}}" "-type=s" "-targets=offload-host-powerpc64le--linux,offload-device-powerpc64le-ibm-linux-gnu,offload-device-x86_64-pc-linux-gnu" "-inputs=[[AAASM:.+]].s" "-outputs=[[AAHASM:.+]].s,[[AAT1ASM:.+]].s,[[AAT2ASM:.+]].s" "-unbundle"
// CHK-COMMANDS-SEP-DAG:    clang-offload-bundler{{.*}}" "-type=i" "-targets=offload-host-powerpc64le--linux,offload-device-powerpc64le-ibm-linux-gnu,offload-device-x86_64-pc-linux-gnu" "-inputs=[[BBPP:.+]].c" "-outputs=[[BBHPP:.+]].i,[[BBT1PP:.+]].i,[[BBT2PP:.+]].i" "-unbundle"

// CHK-COMMANDS-SEP-ST-DAG:    clang-offload-bundler{{.*}}" "-type=s" "-targets=offload-host-powerpc64le--linux,offload-device-powerpc64le-ibm-linux-gnu,offload-device-x86_64-pc-linux-gnu" "-inputs=[[AAASM:.+]].s" "-outputs=[[AAHASM:.+]].s,[[AAT1ASM:.+]].s,[[AAT2ASM:.+]].s" "-unbundle"
// CHK-COMMANDS-SEP-ST-DAG:    clang-offload-bundler{{.*}}" "-type=i" "-targets=offload-host-powerpc64le--linux,offload-device-powerpc64le-ibm-linux-gnu,offload-device-x86_64-pc-linux-gnu" "-inputs=[[BBPP:.+]].c" "-outputs=[[BBHPP:.+]].i,[[BBT1PP:.+]].i,[[BBT2PP:.+]].i" "-unbundle"

// Create 1st bundle.
// CHK-COMMANDS-SEP-DAG:    clang{{.*}}" "-cc1as" "-triple" "powerpc64le--linux" "-filetype" "obj" {{.*}}"-o" "[[AAHOBJ:.+]].o" "[[AAHASM]].s"
// CHK-COMMANDS-SEP-DAG:    clang{{.*}}" "-cc1as" "-triple" "powerpc64le-ibm-linux-gnu" "-filetype" "obj" {{.*}}"-o" "[[AAT1OBJ:.+]].o" "[[AAT1ASM]].s"
// CHK-COMMANDS-SEP-DAG:    clang{{.*}}" "-cc1as" "-triple" "x86_64-pc-linux-gnu" "-filetype" "obj" {{.*}}"-o" "[[AAT2OBJ:.+]].o" "[[AAT2ASM]].s"
// CHK-COMMANDS-SEP-DAG:    clang-offload-bundler{{.*}}" "-type=o" "-targets=offload-host-powerpc64le--linux,offload-device-powerpc64le-ibm-linux-gnu,offload-device-x86_64-pc-linux-gnu" "-outputs=[[AAOBJ:.+]].o" "-inputs=[[AAHOBJ]].o,[[AAT1OBJ]].o,[[AAT2OBJ]].o"

// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1as" "-triple" "powerpc64le--linux" "-filetype" "obj" {{.*}}"-o" "[[AAHOBJ:.+]].o" "[[AAHASM]].s"
// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1as" "-triple" "powerpc64le-ibm-linux-gnu" "-filetype" "obj" {{.*}}"-o" "[[AAT1OBJ:.+]].o" "[[AAT1ASM]].s"
// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1as" "-triple" "x86_64-pc-linux-gnu" "-filetype" "obj" {{.*}}"-o" "[[AAT2OBJ:.+]].o" "[[AAT2ASM]].s"
// CHK-COMMANDS-SEP-ST-DAG:    clang-offload-bundler{{.*}}" "-type=o" "-targets=offload-host-powerpc64le--linux,offload-device-powerpc64le-ibm-linux-gnu,offload-device-x86_64-pc-linux-gnu" "-outputs=[[AAOBJ:.+]].o" "-inputs=[[AAHOBJ]].o,[[AAT1OBJ]].o,[[AAT2OBJ]].o"

// Create 2nd bundle.
// CHK-COMMANDS-SEP-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le--linux" {{.*}}"-emit-llvm-bc"{{.*}}"-fopenmp" {{.*}}"-o" "[[BBHBC:.+]].bc" "-x" "cpp-output" "[[BBHPP]].i" "-fopenmp-targets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu"
// CHK-COMMANDS-SEP-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le--linux" {{.*}}"-emit-obj" {{.*}}"-fopenmp" {{.*}}"-o" "[[BBHOBJ:.+]].o" "-x" "ir" "[[BBHBC]].bc"

// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le--linux" {{.*}}"-emit-llvm-bc"{{.*}}"-fopenmp" {{.*}}"-o" "[[BBHBC:.+]].bc" "-x" "cpp-output" "[[BBHPP]].i" "-fopenmp-targets=powerpc64le-ibm-linux-gnu,x86_64-pc-linux-gnu"
// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le--linux" {{.*}}"-S" {{.*}}"-fopenmp" {{.*}}"-o" "[[BBHASM:.+]].s" "-x" "ir" "[[BBHBC]].bc"
// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1as" "-triple" "powerpc64le--linux" "-filetype" "obj" {{.*}}"-o" "[[BBHOBJ:.+]].o" "[[BBHASM]].s"

// CHK-COMMANDS-SEP-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-emit-llvm-bc" {{.*}}"-fopenmp" {{.*}}"-o" "[[BBT1BC:.+]].bc" "-x" "cpp-output" "[[BBT1PP]].i" "-fopenmp-is-device" "-fopenmp-host-ir-file-path" "[[BBHBC]].bc"
// CHK-COMMANDS-SEP-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-emit-obj" {{.*}}"-fopenmp" {{.*}}"-o" "[[BBT1OBJ:.+]].o" "-x" "ir" "[[BBT1BC]].bc"

// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-emit-llvm-bc" {{.*}}"-fopenmp" {{.*}}"-o" "[[BBT1BC:.+]].bc" "-x" "cpp-output" "[[BBT1PP]].i" "-fopenmp-is-device" "-fopenmp-host-ir-file-path" "[[BBHBC]].bc"
// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "powerpc64le-ibm-linux-gnu" {{.*}}"-S" {{.*}}"-fopenmp" {{.*}}"-o" "[[BBT1ASM:.+]].s" "-x" "ir" "[[BBT1BC]].bc"
// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1as" "-triple" "powerpc64le-ibm-linux-gnu" "-filetype" "obj" {{.*}}"-o" "[[BBT1OBJ:.+]].o" "[[BBT1ASM]].s"

// CHK-COMMANDS-SEP-DAG:    clang{{.*}}" "-cc1" "-triple" "x86_64-pc-linux-gnu" {{.*}}"-emit-llvm-bc" {{.*}}"-fopenmp" {{.*}}"-o" "[[BBT2BC:.+]].bc" "-x" "cpp-output" "[[BBT2PP]].i" "-fopenmp-is-device" "-fopenmp-host-ir-file-path" "[[BBHBC]].bc"
// CHK-COMMANDS-SEP-DAG:    clang{{.*}}" "-cc1" "-triple" "x86_64-pc-linux-gnu" {{.*}}"-emit-obj" {{.*}}"-fopenmp" {{.*}}"-o" "[[BBT2OBJ:.+]].o" "-x" "ir" "[[BBT2BC]].bc"

// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "x86_64-pc-linux-gnu" {{.*}}"-emit-llvm-bc" {{.*}}"-fopenmp" {{.*}}"-o" "[[BBT2BC:.+]].bc" "-x" "cpp-output" "[[BBT2PP]].i" "-fopenmp-is-device" "-fopenmp-host-ir-file-path" "[[BBHBC]].bc"
// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1" "-triple" "x86_64-pc-linux-gnu" {{.*}}"-S" {{.*}}"-fopenmp" {{.*}}"-o" "[[BBT2ASM:.+]].s" "-x" "ir" "[[BBT2BC]].bc"
// CHK-COMMANDS-SEP-ST-DAG:    clang{{.*}}" "-cc1as" "-triple" "x86_64-pc-linux-gnu" "-filetype" "obj" {{.*}}"-o" "[[BBT2OBJ:.+]].o" "[[BBT2ASM]].s"

// CHK-COMMANDS-SEP-DAG:     clang-offload-bundler{{.*}}" "-type=o" "-targets=offload-host-powerpc64le--linux,offload-device-powerpc64le-ibm-linux-gnu,offload-device-x86_64-pc-linux-gnu" "-outputs=[[BBOBJ:.+]].o" "-inputs=[[BBHOBJ]].o,[[BBT1OBJ]].o,[[BBT2OBJ]].o"
// CHK-COMMANDS-SEP-ST-DAG:  clang-offload-bundler{{.*}}" "-type=o" "-targets=offload-host-powerpc64le--linux,offload-device-powerpc64le-ibm-linux-gnu,offload-device-x86_64-pc-linux-gnu" "-outputs=[[BBOBJ:.+]].o" "-inputs=[[BBHOBJ]].o,[[BBT1OBJ]].o,[[BBT2OBJ]].o"



