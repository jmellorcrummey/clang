//===--- Action.cpp - Abstract compilation steps --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Action.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include <cassert>
using namespace clang::driver;
using namespace llvm::opt;

Action::~Action() {}

const char *Action::getClassName(ActionClass AC) {
  switch (AC) {
  case InputClass: return "input";
  case BindArchClass: return "bind-arch";
  case OffloadClass: return "offload";
  case PreprocessJobClass: return "preprocessor";
  case PrecompileJobClass: return "precompiler";
  case AnalyzeJobClass: return "analyzer";
  case MigrateJobClass: return "migrator";
  case CompileJobClass: return "compiler";
  case BackendJobClass: return "backend";
  case AssembleJobClass: return "assembler";
  case LinkJobClass: return "linker";
  case LipoJobClass: return "lipo";
  case DsymutilJobClass: return "dsymutil";
  case VerifyDebugInfoJobClass: return "verify-debug-info";
  case VerifyPCHJobClass: return "verify-pch";
  }

  llvm_unreachable("invalid class");
}

void Action::propagateDeviceOffloadInfo(OffloadKind OKind,
                                        const char *OArch) const {
  // Offload action set its own kinds on their dependences.
  if (Kind == OffloadClass)
    return;

  assert(
      (OffloadingDeviceKind == OKind || OffloadingDeviceKind == OFFLOAD_None) &&
      "Setting device kind to a different device??");
  assert(!OffloadingHostKind && "Setting a device kind in a host action??");
  OffloadingDeviceKind = OKind;
  OffloadingArch = OArch;

  for (auto *A : Inputs)
    A->propagateDeviceOffloadInfo(OffloadingDeviceKind, OArch);
}

void Action::propagateHostOffloadInfo(unsigned OKinds,
                                      const char *OArch) const {
  // Offload action set its own kinds on their dependences.
  if (Kind == OffloadClass)
    return;

  assert(OffloadingDeviceKind == OFFLOAD_None &&
         "Setting a host kind in a device action.");
  OffloadingHostKind |= OKinds;
  OffloadingArch = OArch;

  for (auto *A : Inputs)
    A->propagateHostOffloadInfo(OffloadingHostKind, OArch);
}

void Action::propagateOffloadInfo(const Action *A) const {
  if (unsigned HK = A->getOffloadingHostKinds())
    propagateHostOffloadInfo(HK, A->getOffloadingArch());
  else
    propagateDeviceOffloadInfo(A->getOffloadingDeviceKind(),
                               A->getOffloadingArch());
}

std::string Action::getOffloadingKindPrefix() const {
  switch (OffloadingDeviceKind) {
  case OFFLOAD_None:
    break;
  case OFFLOAD_CUDA:
    return "device-cuda";
    // Add other programming models here.
  }

  if (!OffloadingHostKind)
    return "";

  std::string Res("host");
  if (OffloadingHostKind & OFFLOAD_CUDA)
    Res += "-cuda";
  // Add other programming models here.

  return Res;
}

std::string Action::getOffloadingFileNamePrefix(const ToolChain *TC) const {
  // A file prefix is only generated for device actions and consists of the
  // offload kind and triple.
  if (!OffloadingDeviceKind)
    return "";

  std::string Res("-");
  Res += getOffloadingKindPrefix();
  Res += "-";
  Res += TC->getTriple().normalize();
  return Res;
}

void InputAction::anchor() {}

InputAction::InputAction(const Arg &_Input, types::ID _Type)
  : Action(InputClass, _Type), Input(_Input) {
}

void BindArchAction::anchor() {}

BindArchAction::BindArchAction(Action *Input, const char *_ArchName)
    : Action(BindArchClass, Input), ArchName(_ArchName) {}

void OffloadAction::anchor() {}

OffloadAction::OffloadAction(const HostDependence &HDep)
    : Action(OffloadClass, HDep.getAction()), HostTC(HDep.getToolChain()) {
  OffloadingArch = HDep.getBoundArch();
  OffloadingHostKind = HDep.getOffloadKinds();
  HDep.getAction()->propagateHostOffloadInfo(HDep.getOffloadKinds(),
                                             HDep.getBoundArch());
};

OffloadAction::OffloadAction(const DeviceDependences &DDeps, types::ID Ty)
    : Action(OffloadClass, DDeps.getActions(), Ty), HostTC(nullptr),
      DevToolChains(DDeps.getToolChains()) {
  auto &OKinds = DDeps.getOffloadKinds();
  auto &BArchs = DDeps.getBoundArchs();

  // If we have a single dependency, inherit the offloading info from it.
  if (OKinds.size() == 1) {
    OffloadingDeviceKind = OKinds.front();
    OffloadingArch = BArchs.front();
  }
  // Propagate info to the dependencies.
  for (unsigned i = 0; i < getInputs().size(); ++i)
    getInputs()[i]->propagateDeviceOffloadInfo(OKinds[i], BArchs[i]);
}

OffloadAction::OffloadAction(const HostDependence &HDep,
                             const DeviceDependences &DDeps)
    : Action(OffloadClass, HDep.getAction()), HostTC(HDep.getToolChain()),
      DevToolChains(DDeps.getToolChains()) {
  // We use the kinds of the host dependence for this action.
  OffloadingArch = HDep.getBoundArch();
  OffloadingHostKind = HDep.getOffloadKinds();
  HDep.getAction()->propagateHostOffloadInfo(HDep.getOffloadKinds(),
                                             HDep.getBoundArch());

  // Add device inputs and propagate info to the device actions.
  for (unsigned i = 0; i < DDeps.getActions().size(); ++i) {
    auto *A = DDeps.getActions()[i];
    // Skip actions of empty dependences.
    if (!A)
      continue;
    getInputs().push_back(A);
    A->propagateDeviceOffloadInfo(DDeps.getOffloadKinds()[i],
                                  DDeps.getBoundArchs()[i]);
  }
}

void OffloadAction::doOnHostDependence(const OffloadActionWorkTy &Work) const {
  if (!HostTC)
    return;
  auto *A = getInputs().front();
  Work(A, HostTC, A->getOffloadingArch());
}

void OffloadAction::doOnEachDeviceDependence(
    const OffloadActionWorkTy &Work) const {
  auto I = getInputs().begin();
  auto E = getInputs().end();
  if (I == E)
    return;

  // Skip host action
  if (HostTC)
    ++I;

  auto TI = DevToolChains.begin();
  for (; I != E; ++I)
    Work(*I, *TI, (*I)->getOffloadingArch());
}

void OffloadAction::doOnEachDependence(const OffloadActionWorkTy &Work) const {
  doOnHostDependence(Work);
  doOnEachDeviceDependence(Work);
}

Action *OffloadAction::getHostDependence() const {
  return HostTC ? getInputs().front() : nullptr;
}

Action *OffloadAction::getSingleDeviceDependence() const {
  return (!HostTC && getInputs().size() == 1) ? getInputs().front() : nullptr;
}

void OffloadAction::DeviceDependences::add(Action *A, const ToolChain *TC,
                                           const char *BoundArch,
                                           OffloadKind OKind) {
  AL.push_back(A);
  TCL.push_back(TC);
  BAL.push_back(BoundArch);
  KL.push_back(OKind);
}

OffloadAction::HostDependence::HostDependence(Action *A, const ToolChain *TC,
                                              const char *BoundArch,
                                              const DeviceDependences &DDeps)
    : A(A), TC(TC), BoundArch(BoundArch), OffloadKinds(0u) {
  for (auto K : DDeps.getOffloadKinds())
    OffloadKinds |= K;
}

void JobAction::anchor() {}

JobAction::JobAction(ActionClass Kind, Action *Input, types::ID Type)
    : Action(Kind, Input, Type) {}

JobAction::JobAction(ActionClass Kind, const ActionList &Inputs, types::ID Type)
  : Action(Kind, Inputs, Type) {
}

void PreprocessJobAction::anchor() {}

PreprocessJobAction::PreprocessJobAction(Action *Input, types::ID OutputType)
    : JobAction(PreprocessJobClass, Input, OutputType) {}

void PrecompileJobAction::anchor() {}

PrecompileJobAction::PrecompileJobAction(Action *Input, types::ID OutputType)
    : JobAction(PrecompileJobClass, Input, OutputType) {}

void AnalyzeJobAction::anchor() {}

AnalyzeJobAction::AnalyzeJobAction(Action *Input, types::ID OutputType)
    : JobAction(AnalyzeJobClass, Input, OutputType) {}

void MigrateJobAction::anchor() {}

MigrateJobAction::MigrateJobAction(Action *Input, types::ID OutputType)
    : JobAction(MigrateJobClass, Input, OutputType) {}

void CompileJobAction::anchor() {}

CompileJobAction::CompileJobAction(Action *Input, types::ID OutputType)
    : JobAction(CompileJobClass, Input, OutputType) {}

void BackendJobAction::anchor() {}

BackendJobAction::BackendJobAction(Action *Input, types::ID OutputType)
    : JobAction(BackendJobClass, Input, OutputType) {}

void AssembleJobAction::anchor() {}

AssembleJobAction::AssembleJobAction(Action *Input, types::ID OutputType)
    : JobAction(AssembleJobClass, Input, OutputType) {}

void LinkJobAction::anchor() {}

LinkJobAction::LinkJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(LinkJobClass, Inputs, Type) {
}

void LipoJobAction::anchor() {}

LipoJobAction::LipoJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(LipoJobClass, Inputs, Type) {
}

void DsymutilJobAction::anchor() {}

DsymutilJobAction::DsymutilJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(DsymutilJobClass, Inputs, Type) {
}

void VerifyJobAction::anchor() {}

VerifyJobAction::VerifyJobAction(ActionClass Kind, Action *Input,
                                 types::ID Type)
    : JobAction(Kind, Input, Type) {
  assert((Kind == VerifyDebugInfoJobClass || Kind == VerifyPCHJobClass) &&
         "ActionClass is not a valid VerifyJobAction");
}

void VerifyDebugInfoJobAction::anchor() {}

VerifyDebugInfoJobAction::VerifyDebugInfoJobAction(Action *Input,
                                                   types::ID Type)
    : VerifyJobAction(VerifyDebugInfoJobClass, Input, Type) {}

void VerifyPCHJobAction::anchor() {}

VerifyPCHJobAction::VerifyPCHJobAction(Action *Input, types::ID Type)
    : VerifyJobAction(VerifyPCHJobClass, Input, Type) {}
