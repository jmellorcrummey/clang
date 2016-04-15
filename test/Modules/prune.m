// Test the automatic pruning of module cache entries.
#ifdef IMPORT_DEPENDS_ON_MODULE
@import DependsOnModule;
#else
@import Module;
#endif

// We need 'touch' and 'find' for this test to work.
// REQUIRES: shell

// RUN: /bin/true

// Clear out the module cache
// RxUN: rm -rf %t
// Run Clang twice so we end up creating the timestamp file (the second time).
// RxUN: %clang_cc1 -DIMPORT_DEPENDS_ON_MODULE -fmodules-ignore-macro=DIMPORT_DEPENDS_ON_MODULE -fmodules -fimplicit-module-maps -F %S/Inputs -fmodules-cache-path=%t %s -verify
// RxUN: %clang_cc1 -DIMPORT_DEPENDS_ON_MODULE -fmodules-ignore-macro=DIMPORT_DEPENDS_ON_MODULE -fmodules -fimplicit-module-maps -F %S/Inputs -fmodules-cache-path=%t %s -verify
// RxUN: ls %t | grep modules.timestamp
// RxUN: ls -R %t | grep ^Module.*pcm
// RxUN: ls -R %t | grep DependsOnModule.*pcm

// Set the timestamp back more than two days. We should try to prune,
// but nothing gets pruned because the module files are new enough.
// RxUN: touch -m -a -t 201101010000 %t/modules.timestamp 
// RxUN: %clang_cc1 -fmodules -fimplicit-module-maps -F %S/Inputs -fmodules-cache-path=%t -fmodules -fmodules-prune-interval=172800 -fmodules-prune-after=345600 %s -verify
// RxUN: ls %t | grep modules.timestamp
// RxUN: ls -R %t | grep ^Module.*pcm
// RxUN: ls -R %t | grep DependsOnModule.*pcm

// Set the DependsOnModule access time back more than four days.
// This shouldn't prune anything, because the timestamp has been updated, so
// the pruning mechanism won't fire.
// RxUN: find %t -name DependsOnModule*.pcm | xargs touch -a -t 201101010000
// RxUN: %clang_cc1 -fmodules -fimplicit-module-maps -F %S/Inputs -fmodules-cache-path=%t -fmodules -fmodules-prune-interval=172800 -fmodules-prune-after=345600 %s -verify
// RxUN: ls %t | grep modules.timestamp
// RxUN: ls -R %t | grep ^Module.*pcm
// RxUN: ls -R %t | grep DependsOnModule.*pcm

// Set both timestamp and DependsOnModule.pcm back beyond the cutoff.
// This should trigger pruning, which will remove DependsOnModule but not Module.
// RxUN: touch -m -a -t 201101010000 %t/modules.timestamp 
// RxUN: find %t -name DependsOnModule*.pcm | xargs touch -a -t 201101010000
// RxUN: %clang_cc1 -fmodules -fimplicit-module-maps -F %S/Inputs -fmodules-cache-path=%t -fmodules -fmodules-prune-interval=172800 -fmodules-prune-after=345600 %s -verify
// RxUN: ls %t | grep modules.timestamp
// RxUN: ls -R %t | grep ^Module.*pcm
// RxUN: ls -R %t | not grep DependsOnModule.*pcm

// expected-no-diagnostics
