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

namespace clang{
namespace CodeGen{

class CGOpenMPRuntimeNVPTX : public CGOpenMPRuntime {
public:
  explicit CGOpenMPRuntimeNVPTX(CodeGenModule &CGM) : CGOpenMPRuntime(CGM) {}
};


CGOpenMPRuntime *CGOpenMPRuntime::createCGOpenMPRuntimeNVPTX(CodeGenModule &CGM){
  return new CGOpenMPRuntimeNVPTX(CGM);
}
} // CodeGen namespace.
} // clang namespace.
