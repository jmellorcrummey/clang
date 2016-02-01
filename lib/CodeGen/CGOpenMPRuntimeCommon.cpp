//===--- CGOpenMPRuntimeCommon.cpp - Helpers for OpenMP Runtime Codegen ----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides several common helper classes and functions useful for the
// default and target-specialized OpenMP runtime code generation.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntimeCommon.h"
#include "CodeGenFunction.h"

namespace clang {
namespace CodeGen {
namespace CGOpenMPCommon {

LValue emitLoadOfPointerLValue(CodeGenFunction &CGF, Address PtrAddr,
                                      QualType Ty) {
  AlignmentSource Source;
  CharUnits Align = CGF.getNaturalPointeeTypeAlignment(Ty, &Source);
  return CGF.MakeAddrLValue(Address(CGF.Builder.CreateLoad(PtrAddr), Align),
                            Ty->getPointeeType(), Source);
}

LValue CGOpenMPRegionInfo::getThreadIDVariableLValue(CodeGenFunction &CGF) {
  return emitLoadOfPointerLValue(CGF,
                                 CGF.GetAddrOfLocalVar(getThreadIDVariable()),
                                 getThreadIDVariable()->getType());
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

LValue CGOpenMPTaskOutlinedRegionInfo::getThreadIDVariableLValue(
    CodeGenFunction &CGF) {
  return CGF.MakeAddrLValue(CGF.GetAddrOfLocalVar(getThreadIDVariable()),
                            getThreadIDVariable()->getType(),
                            AlignmentSource::Decl);
}

/// \brief Emits code for OpenMP 'if' clause using specified \a CodeGen
/// function. Here is the logic:
/// if (Cond) {
///   ThenGen();
/// } else {
///   ElseGen();
/// }
void emitOMPIfClause(CodeGenFunction &CGF, const Expr *Cond,
                            const RegionCodeGenTy &ThenGen,
                            const RegionCodeGenTy &ElseGen) {
  CodeGenFunction::LexicalScope ConditionScope(CGF, Cond->getSourceRange());

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm of the if/else.
  bool CondConstant;
  if (CGF.ConstantFoldsToSimpleInteger(Cond, CondConstant)) {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    if (CondConstant) {
      ThenGen(CGF);
    } else {
      ElseGen(CGF);
    }
    return;
  }

  // Otherwise, the condition did not fold, or we couldn't elide it.  Just
  // emit the conditional branch.
  auto ThenBlock = CGF.createBasicBlock("omp_if.then");
  auto ElseBlock = CGF.createBasicBlock("omp_if.else");
  auto ContBlock = CGF.createBasicBlock("omp_if.end");
  CGF.EmitBranchOnBoolExpr(Cond, ThenBlock, ElseBlock, /*TrueCount=*/0);

  // Emit the 'then' code.
  CGF.EmitBlock(ThenBlock);
  {
    CodeGenFunction::RunCleanupsScope ThenScope(CGF);
    ThenGen(CGF);
  }
  CGF.EmitBranch(ContBlock);
  // Emit the 'else' code if present.
  {
    // There is no need to emit line number for unconditional branch.
    auto NL = ApplyDebugLocation::CreateEmpty(CGF);
    CGF.EmitBlock(ElseBlock);
  }
  {
    CodeGenFunction::RunCleanupsScope ThenScope(CGF);
    ElseGen(CGF);
  }
  {
    // There is no need to emit line number for unconditional branch.
    auto NL = ApplyDebugLocation::CreateEmpty(CGF);
    CGF.EmitBranch(ContBlock);
  }
  // Emit the continuation block for code after the if.
  CGF.EmitBlock(ContBlock, /*IsFinished=*/true);
}

} // CGOpenMPCommon namespace.
} // CodeGen namespace.
} // clang namespace.
