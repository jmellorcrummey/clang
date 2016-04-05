//===--- Action.h - Abstract compilation steps ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_ACTION_H
#define LLVM_CLANG_DRIVER_ACTION_H

#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class StringRef;

namespace opt {
  class Arg;
}
}

namespace clang {
namespace driver {

class ToolChain;

/// Action - Represent an abstract compilation step to perform.
///
/// An action represents an edge in the compilation graph; typically
/// it is a job to transform an input using some tool.
///
/// The current driver is hard wired to expect actions which produce a
/// single primary output, at least in terms of controlling the
/// compilation. Actions can produce auxiliary files, but can only
/// produce a single output to feed into subsequent actions.
///
/// Actions are usually owned by a Compilation, which creates new
/// actions via MakeAction().
class Action {
public:
  typedef ActionList::size_type size_type;
  typedef ActionList::iterator input_iterator;
  typedef ActionList::const_iterator input_const_iterator;
  typedef llvm::iterator_range<input_iterator> input_range;
  typedef llvm::iterator_range<input_const_iterator> input_const_range;

  enum ActionClass {
    InputClass = 0,
    BindArchClass,
    OffloadClass,
    PreprocessJobClass,
    PrecompileJobClass,
    AnalyzeJobClass,
    MigrateJobClass,
    CompileJobClass,
    BackendJobClass,
    AssembleJobClass,
    LinkJobClass,
    LipoJobClass,
    DsymutilJobClass,
    VerifyDebugInfoJobClass,
    VerifyPCHJobClass,

    JobClassFirst=PreprocessJobClass,
    JobClassLast=VerifyPCHJobClass
  };

  // The offloading kind determines if this action is binded to a particular
  // programming model. Each entry reserves one bit. We also have a special kind
  // to designate the host offloading tool chain.
  enum OffloadKind {
    OFK_None = 0x00,
    // The host offloading tool chain.
    OFK_Host = 0x01,
    // The device offloading tool chains - one bit for each programming model.
    OFK_Cuda = 0x02,
  };

  static const char *getClassName(ActionClass AC);

private:
  ActionClass Kind;

  /// The output type of this action.
  types::ID Type;

  ActionList Inputs;

protected:
  /// Offload information. It has to be mutable as it needs to be adjusted if
  /// actions are integrated.
  /// \brief Multiple programming models may be supported simultaneously by the
  /// same host. Therefore, the host offloading kind is a combination of kinds.
  mutable unsigned OffloadingHostKind;
  /// \brief Offloading kind of the device.
  mutable OffloadKind OffloadingDeviceKind;
  /// \brief The Offloading architecture associated with this action.
  mutable const char *OffloadingArch;

  Action(ActionClass Kind, types::ID Type) : Action(Kind, ActionList(), Type) {}
  Action(ActionClass Kind, Action *Input, types::ID Type)
      : Action(Kind, ActionList({Input}), Type) {}
  Action(ActionClass Kind, Action *Input)
      : Action(Kind, ActionList({Input}), Input->getType()) {}
  Action(ActionClass Kind, const ActionList &Inputs, types::ID Type)
      : Kind(Kind), Type(Type), Inputs(Inputs), OffloadingHostKind(0u),
        OffloadingDeviceKind(OFK_None), OffloadingArch(nullptr) {}

public:
  virtual ~Action();

  const char *getClassName() const { return Action::getClassName(getKind()); }

  ActionClass getKind() const { return Kind; }
  types::ID getType() const { return Type; }

  ActionList &getInputs() { return Inputs; }
  const ActionList &getInputs() const { return Inputs; }

  size_type size() const { return Inputs.size(); }

  input_iterator input_begin() { return Inputs.begin(); }
  input_iterator input_end() { return Inputs.end(); }
  input_range inputs() { return input_range(input_begin(), input_end()); }
  input_const_iterator input_begin() const { return Inputs.begin(); }
  input_const_iterator input_end() const { return Inputs.end(); }
  input_const_range inputs() const {
    return input_const_range(input_begin(), input_end());
  }

  std::string getOffloadingKindPrefix() const;
  std::string getOffloadingFileNamePrefix(const ToolChain *TC) const;

  /// \brief Set the device offload info of this action and propagate it to its
  /// dependences.
  void propagateDeviceOffloadInfo(OffloadKind OKind, const char *OArch) const;
  /// \brief Append the host offload info of this action and propagate it to its
  /// dependences.
  void propagateHostOffloadInfo(unsigned OKinds, const char *OArch) const;
  /// \brief Set the offload info of this action to be the same as the provided
  /// action, and propagate it to its dependences.
  void propagateOffloadInfo(const Action *A) const;

  unsigned getOffloadingHostKinds() const { return OffloadingHostKind; }
  OffloadKind getOffloadingDeviceKind() const { return OffloadingDeviceKind; }
  const char *getOffloadingArch() const { return OffloadingArch; }

  /// \brief Check if this action have any offload kinds. Note that host offload
  /// kinds are only set if the action is a dependence to an host offload
  /// action.
  bool isHostOffloading(OffloadKind OKind) const {
    return OffloadingHostKind & OKind;
  }
  bool isDeviceOffloading(OffloadKind OKind) const {
    return OffloadingDeviceKind == OKind;
  }
  bool isOffloading(OffloadKind OKind) const {
    return isHostOffloading(OKind) || isDeviceOffloading(OKind);
  }
};

class InputAction : public Action {
  virtual void anchor();
  const llvm::opt::Arg &Input;

public:
  InputAction(const llvm::opt::Arg &Input, types::ID Type);

  const llvm::opt::Arg &getInputArg() const { return Input; }

  static bool classof(const Action *A) {
    return A->getKind() == InputClass;
  }
};

class BindArchAction : public Action {
  virtual void anchor();
  /// The architecture to bind, or 0 if the default architecture
  /// should be bound.
  const char *ArchName;

public:
  BindArchAction(Action *Input, const char *ArchName);

  const char *getArchName() const { return ArchName; }

  static bool classof(const Action *A) {
    return A->getKind() == BindArchClass;
  }
};

/// \brief An offload action combines host or/and device actions according to
/// the programming model implementation needs and propagates the offloading
/// kind to its dependences.
class OffloadAction : public Action {
  virtual void anchor();
public:
  /// \brief Type used to communicate device actions. It associates bound
  /// architecture, toolchain, and offload kind to each action.
  class DeviceDependences {
  public:
    typedef SmallVector<const ToolChain *, 3> ToolChainList;
    typedef SmallVector<const char *, 3> BoundArchList;
    typedef SmallVector<OffloadKind, 3> OffloadKindList;

  private:
    /// \brief The dependence action.
    ActionList AL;
    /// \brief The offloading toolchains that should be used with the action.
    SmallVector<const ToolChain *, 3> TCL;
    /// \brief The architectures that should be used with this action.
    SmallVector<const char *, 3> BAL;
    /// \brief The offload kind of each dependence.
    SmallVector<OffloadKind, 3> KL;

  public:
    /// \brief Add a action along with the associated toolchain, bound arch, and
    /// offload kind.
    void add(Action *A, const ToolChain *TC, const char *BoundArch,
             OffloadKind OKind);

    /// \brief Get each of the individual arrays.
    const ActionList &getActions() const { return AL; };
    const ToolChainList &getToolChains() const { return TCL; };
    const BoundArchList &getBoundArchs() const { return BAL; };
    const OffloadKindList &getOffloadKinds() const { return KL; };
  };

  /// \brief Type used to communicate host actions. It associates bound
  /// architecture, toolchain, and offload kinds to each action.
  class HostDependence {
    /// \brief The dependence action.
    Action *A;
    /// \brief The offloading toolchain that should be used with the action.
    const ToolChain *TC;
    /// \brief The architectures that should be used with this action.
    const char *BoundArch;
    /// \brief The offload kind of each dependence.
    unsigned OffloadKinds;

  public:
    HostDependence(Action *A, const ToolChain *TC, const char *BoundArch,
                   const unsigned OffloadKinds)
        : A(A), TC(TC), BoundArch(BoundArch), OffloadKinds(OffloadKinds){};
    /// \brief Constructor version that obtains the offload kinds from the
    /// device dependencies.
    HostDependence(Action *A, const ToolChain *TC, const char *BoundArch,
                   const DeviceDependences &DDeps);
    Action *getAction() const { return A; };
    const ToolChain *getToolChain() const { return TC; };
    const char *getBoundArch() const { return BoundArch; };
    unsigned getOffloadKinds() const { return OffloadKinds; };
  };

  typedef llvm::function_ref<void(Action *, const ToolChain *, const char *)>
      OffloadActionWorkTy;

private:
  /// \brief The offloading toolchain that should be used with the action.
  const ToolChain *HostTC;

  /// \brief The tool chains associated with the list of actions.
  DeviceDependences::ToolChainList DevToolChains;

public:
  OffloadAction(const HostDependence &HDep);
  OffloadAction(const DeviceDependences &DDeps, types::ID Ty);
  OffloadAction(const HostDependence &HDep, const DeviceDependences &DDeps);

  /// \brief Execute the work specified in \a Work on the host dependence.
  void doOnHostDependence(const OffloadActionWorkTy &Work) const;

  /// \brief Execute the work specified in \a Work on each device dependence.
  void doOnEachDeviceDependence(const OffloadActionWorkTy &Work) const;

  /// \brief Execute the work specified in \a Work on each dependence.
  void doOnEachDependence(const OffloadActionWorkTy &Work) const;

  /// \brief Return the host dependence of this action, or null if we don't have
  /// any.
  Action *getHostDependence() const;

  /// \brief Return the single device dependence of this action, or null if we
  /// don't have one or we have more than one.
  Action *getSingleDeviceDependence() const;

  static bool classof(const Action *A) { return A->getKind() == OffloadClass; }
};

class JobAction : public Action {
  virtual void anchor();
protected:
  JobAction(ActionClass Kind, Action *Input, types::ID Type);
  JobAction(ActionClass Kind, const ActionList &Inputs, types::ID Type);

public:
  static bool classof(const Action *A) {
    return (A->getKind() >= JobClassFirst &&
            A->getKind() <= JobClassLast);
  }
};

class PreprocessJobAction : public JobAction {
  void anchor() override;
public:
  PreprocessJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == PreprocessJobClass;
  }
};

class PrecompileJobAction : public JobAction {
  void anchor() override;
public:
  PrecompileJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == PrecompileJobClass;
  }
};

class AnalyzeJobAction : public JobAction {
  void anchor() override;
public:
  AnalyzeJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == AnalyzeJobClass;
  }
};

class MigrateJobAction : public JobAction {
  void anchor() override;
public:
  MigrateJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == MigrateJobClass;
  }
};

class CompileJobAction : public JobAction {
  void anchor() override;
public:
  CompileJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == CompileJobClass;
  }
};

class BackendJobAction : public JobAction {
  void anchor() override;
public:
  BackendJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == BackendJobClass;
  }
};

class AssembleJobAction : public JobAction {
  void anchor() override;
public:
  AssembleJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == AssembleJobClass;
  }
};

class LinkJobAction : public JobAction {
  void anchor() override;
public:
  LinkJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == LinkJobClass;
  }
};

class LipoJobAction : public JobAction {
  void anchor() override;
public:
  LipoJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == LipoJobClass;
  }
};

class DsymutilJobAction : public JobAction {
  void anchor() override;
public:
  DsymutilJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == DsymutilJobClass;
  }
};

class VerifyJobAction : public JobAction {
  void anchor() override;
public:
  VerifyJobAction(ActionClass Kind, Action *Input, types::ID Type);
  static bool classof(const Action *A) {
    return A->getKind() == VerifyDebugInfoJobClass ||
           A->getKind() == VerifyPCHJobClass;
  }
};

class VerifyDebugInfoJobAction : public VerifyJobAction {
  void anchor() override;
public:
  VerifyDebugInfoJobAction(Action *Input, types::ID Type);
  static bool classof(const Action *A) {
    return A->getKind() == VerifyDebugInfoJobClass;
  }
};

class VerifyPCHJobAction : public VerifyJobAction {
  void anchor() override;
public:
  VerifyPCHJobAction(Action *Input, types::ID Type);
  static bool classof(const Action *A) {
    return A->getKind() == VerifyPCHJobClass;
  }
};

} // end namespace driver
} // end namespace clang

#endif
