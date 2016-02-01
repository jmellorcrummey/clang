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

#include "CGOpenMPRuntime.h"
#include "CGOpenMPRuntimeCommon.h"

namespace clang{
namespace CodeGen{

class CGOpenMPRuntimeNVPTX : public CGOpenMPRuntime {
public:
  explicit CGOpenMPRuntimeNVPTX(CodeGenModule &CGM) : CGOpenMPRuntime(CGM) {
    if (!CGM.getLangOpts().OpenMPIsDevice)
      llvm_unreachable("OpenMP NVPTX is only prepared to deal with device code.");
  }
};

namespace CGOpenMPCommon{
  CGOpenMPRuntime *createCGOpenMPRuntimeNVPTX(CodeGenModule &CGM){
    return new CGOpenMPRuntimeNVPTX(CGM);
  }
} // CGOpenMPCommon namespace.
} // CodeGen namespace.
} // clang namespace.
