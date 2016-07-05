//===-- clang-offload-bundler/ClangOffloadBundler.cpp - Clang format tool -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a clang-offload-bundler that bundles different
/// files that relate with the same source code but different targets into a
/// single one. Also the implements the opposite functionality, i.e. unbundle
/// files previous created by this tool.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileManager.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    ClangOffloadBundlerCategory("clang-offload-bundler options");

static cl::list<std::string>
    InputFileNames("inputs", cl::CommaSeparated, cl::OneOrMore,
                   cl::desc("[<input file>,...]"),
                   cl::cat(ClangOffloadBundlerCategory));
static cl::list<std::string>
    OutputFileNames("outputs", cl::CommaSeparated, cl::OneOrMore,
                    cl::desc("[<output file>,...]"),
                    cl::cat(ClangOffloadBundlerCategory));
static cl::list<std::string> TargetNames("targets", cl::CommaSeparated,
                                         cl::OneOrMore,
                                         cl::desc("[<target triple>,...]"),
                                         cl::cat(ClangOffloadBundlerCategory));
static cl::opt<std::string>
    FilesType("type", cl::Required,
              cl::desc("Type of the files to be bundled/unbundled.\n"
                       "Current supported types are:\n"
                       "  i   - cpp-output\n"
                       "  ii  - c++-cpp-output\n"
                       "  ll  - llvm\n"
                       "  bc  - llvm-bc\n"
                       "  s   - assembler\n"
                       "  o   - object\n"
                       "  gch - precompiled-header\n"
                       "  ast - clang AST file"),
              cl::cat(ClangOffloadBundlerCategory));
static cl::opt<bool>
    Unbundle("unbundle",
             cl::desc("Unbundle bundled file into several output files.\n"),
             cl::init(false), cl::cat(ClangOffloadBundlerCategory));

/// \brief Magic string that marks the existence of offloading data.
#define OFFLOAD_BUNDLER_MAGIC_STR "__CLANG_OFFLOAD_BUNDLE__"

/// \brief Generic file handler interface.
class FileHandler {
protected:
  /// \brief Update the file handler with information from the header of the
  /// bundled file
  virtual void ReadHeader(MemoryBuffer &Input) = 0;
  /// \brief Read the marker of the next bundled to be read in the file. The
  /// triple of the target associated with that bundled is returned. An empty
  /// string is returned if there are no more bundles to be read.
  virtual StringRef ReadBundleStart(MemoryBuffer &Input) = 0;
  /// \brief Read the marker that closes the current bundle.
  virtual void ReadBundleEnd(MemoryBuffer &Input) = 0;
  /// \brief Read the current bundle and write the result into the stream \a OS.
  virtual void ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) = 0;

  /// \brief Write the header of the bundled file to \a OS based on the
  /// information gathered from \a Inputs.
  virtual void WriteHeader(raw_fd_ostream &OS,
                           ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs) = 0;
  /// \brief Write the marker that initiates a bundle for the triple \a
  /// TargetTriple to \a OS.
  virtual void WriteBundleStart(raw_fd_ostream &OS, StringRef TargetTriple) = 0;
  /// \brief Write the marker that closes a bundle for the triple \a
  /// TargetTriple to \a OS.
  virtual void WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) = 0;
  /// \brief Write the bundle from \a Input into \a OS.
  virtual void WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) = 0;

public:
  FileHandler() {}
  virtual ~FileHandler() {}

  /// \brief Bundle the files. Return true if an error was found.
  bool Bundle() {
    std::error_code EC;

    // Create output file.
    raw_fd_ostream OutputFile(OutputFileNames.front(), EC, sys::fs::F_None);

    if (EC) {
      llvm::errs() << "error: Can't open file " << OutputFileNames.front()
                   << ".\n";
      return true;
    }

    // Open input files.
    std::vector<std::unique_ptr<MemoryBuffer>> InputBuffers(
        InputFileNames.size());

    unsigned Idx = 0;
    for (auto I : InputFileNames) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> CodeOrErr =
          MemoryBuffer::getFileOrSTDIN(I);
      if (std::error_code EC = CodeOrErr.getError()) {
        llvm::errs() << EC.message() << "\n";
        return true;
      }
      InputBuffers[Idx++] = std::move(CodeOrErr.get());
    }

    // Write header.
    WriteHeader(OutputFile, InputBuffers);

    // Write all bundles along with the start/end markers.
    auto Input = InputBuffers.begin();
    for (auto Triple = TargetNames.begin(); Triple < TargetNames.end();
         ++Triple, ++Input) {
      WriteBundleStart(OutputFile, *Triple);
      WriteBundle(OutputFile, *Input->get());
      WriteBundleEnd(OutputFile, *Triple);
    }
    return false;
  }

  // Unbundle the files. Return true if an error was found.
  bool Unbundle() {
    // Open Input file.
    ErrorOr<std::unique_ptr<MemoryBuffer>> CodeOrErr =
        MemoryBuffer::getFileOrSTDIN(InputFileNames.front());
    if (std::error_code EC = CodeOrErr.getError()) {
      llvm::errs() << EC.message() << "\n";
      return true;
    }

    // Read the header of the bundled file.
    MemoryBuffer &Input = *CodeOrErr.get();
    ReadHeader(Input);

    // Create a work list that consist of the map triple/output file.
    StringMap<StringRef> Worklist;
    auto Output = OutputFileNames.begin();
    for (auto Triple = TargetNames.begin(); Triple < TargetNames.end();
         ++Triple, ++Output)
      Worklist[*Triple] = *Output;

    // Read all the bundles that are in the work list, and return an error is a
    // given bundle wasn't found.
    while (!Worklist.empty()) {
      StringRef CurTriple = ReadBundleStart(Input);

      if (CurTriple.empty()) {
        llvm::errs()
            << "error: Unable to find bundles for all requested targets.\n";
        return true;
      }

      auto Output = Worklist.find(CurTriple);
      // The file may have more bundles for other targets.
      if (Output == Worklist.end()) {
        continue;
      }

      // Check if the output file can be opened and copy the bundle to it.
      std::error_code EC;
      raw_fd_ostream OutputFile(Output->second, EC, sys::fs::F_None);
      if (EC) {
        llvm::errs() << "error: Can't open file " << Output->second << ".\n";
        return true;
      }
      ReadBundle(OutputFile, Input);
      ReadBundleEnd(Input);
      Worklist.remove(&*Output);
    }

    return false;
  }
};

// Handler for binary files. The bundled file will have the following format
// (all integers are stored in little-endian format):
//
// "OFFLOAD_BUNDLER_MAGIC_STR" (ASCII encoding of the string)
//
// NumberOfOffloadBundles (8-byte integer)
//
// OffsetOfBundle1 (8-byte integer)
// SizeOfBundle1 (8-byte integer)
// NumberOfBytesInTripleOfBundle1 (8-byte integer)
// TripleOfBundle1 (byte length defined before)
//
// ...
//
// OffsetOfBundleN (8-byte integer)
// SizeOfBundleN (8-byte integer)
// NumberOfBytesInTripleOfBundleN (8-byte integer)
// TripleOfBundleN (byte length defined before)
//
// Bundle1
// ...
// BundleN

/// \brief Read 8-byte integers to/from a buffer in little-endian format.
static uint64_t Read8byteIntegerFromBuffer(StringRef Buffer, size_t pos) {
  uint64_t Res = 0;
  const char *Data = Buffer.data();

  for (unsigned i = 0; i < 8; ++i) {
    Res <<= 8;
    uint64_t Char = (uint64_t)Data[pos + 7 - i];
    Res |= 0xffu & Char;
  }
  return Res;
}

/// \brief Write and write 8-byte integers to/from a buffer in little-endian
/// format.
static void Write8byteIntegerToBuffer(raw_fd_ostream &OS, uint64_t Val) {

  for (unsigned i = 0; i < 8; ++i) {
    char Char = (char)(Val & 0xffu);
    OS.write(&Char, 1);
    Val >>= 8;
  }
}

class BinaryFileHandler : public FileHandler {
  /// \brief Information about the bundles extracted from the header.
  struct BundleInfo {
    /// \brief Size of the bundle.
    uint64_t Size;
    /// \brief Offset at which the bundle starts in the bundled file.
    uint64_t Offset;
    BundleInfo() : Size(0), Offset(0) {}
    BundleInfo(uint64_t Size, uint64_t Offset) : Size(Size), Offset(Offset) {}
  };
  /// Map between a triple and the corresponding bundle information.
  StringMap<BundleInfo> BundlesInfo;

  /// Number of triples read so far.
  size_t ReadTriples;

protected:
  void ReadHeader(MemoryBuffer &Input) {
    StringRef FC = Input.getBuffer();

    // Check if buffer is smaller than magic string.
    size_t ReadChars = sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;
    if (ReadChars > FC.size())
      return;

    // Check if no magic was found.
    StringRef Magic(FC.data(), sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
    if (!Magic.equals(OFFLOAD_BUNDLER_MAGIC_STR))
      return;

    // Read number of bundles.
    if (ReadChars + 8 > FC.size())
      return;

    uint64_t NumberOfBundles = Read8byteIntegerFromBuffer(FC, ReadChars);
    ReadChars += 8;

    // Read bundle offsets, sizes and triples.
    for (uint64_t i = 0; i < NumberOfBundles; ++i) {

      // Read offset.
      if (ReadChars + 8 > FC.size())
        return;

      uint64_t Offset = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read size.
      if (ReadChars + 8 > FC.size())
        return;

      uint64_t Size = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read triple size.
      if (ReadChars + 8 > FC.size())
        return;

      uint64_t TripleSize = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read triple.
      if (ReadChars + TripleSize > FC.size())
        return;

      StringRef Triple(&FC.data()[ReadChars], TripleSize);
      ReadChars += TripleSize;

      // Check if the offset and size make sense.
      if (!Size || !Offset || Offset + Size > FC.size())
        return;

      assert(BundlesInfo.find(Triple) == BundlesInfo.end() &&
             "Triple is duplicated??");
      BundlesInfo[Triple] = BundleInfo(Size, Offset);
    }
  }
  StringRef ReadBundleStart(MemoryBuffer &Input) {
    StringRef CurTriple = TargetNames[ReadTriples];
    return CurTriple;
  }
  void ReadBundleEnd(MemoryBuffer &Input) {
    ++ReadTriples;
    return;
  }
  void ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) {
    StringRef FC = Input.getBuffer();
    StringRef CurTriple = TargetNames[ReadTriples];

    auto BI = BundlesInfo.lookup(CurTriple);
    assert(BI.Size && "No bundle info found!");

    OS.write(&FC.data()[BI.Offset], BI.Size);
  }

  void WriteHeader(raw_fd_ostream &OS,
                   ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs) {
    // Compute size of the header.
    uint64_t HeaderSize = 0;

    HeaderSize += sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;
    HeaderSize += 8; // Number of Bundles

    for (auto &T : TargetNames) {
      HeaderSize += 3 * 8; // Bundle offset, Size of bundle and size of triple.
      HeaderSize += T.size(); // The triple.
    }

    // Write to the buffer the header.
    OS << OFFLOAD_BUNDLER_MAGIC_STR;

    Write8byteIntegerToBuffer(OS, TargetNames.size());

    unsigned Idx = 0;
    for (auto &T : TargetNames) {
      MemoryBuffer &MB = *Inputs[Idx++].get();
      // Bundle offset.
      Write8byteIntegerToBuffer(OS, HeaderSize);
      // Size of the bundle (adds to the next bundle's offset)
      Write8byteIntegerToBuffer(OS, MB.getBufferSize());
      HeaderSize += MB.getBufferSize();
      // Size of the triple
      Write8byteIntegerToBuffer(OS, T.size());
      // Triple
      OS << T;
    }
  }
  void WriteBundleStart(raw_fd_ostream &OS, StringRef TargetTriple) { return; }
  void WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) { return; }
  void WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) {
    OS.write(Input.getBufferStart(), Input.getBufferSize());
    return;
  }

public:
  BinaryFileHandler() : FileHandler(), ReadTriples(0) {}
  ~BinaryFileHandler() {}
};

// Handler for text files. The bundled file will have the following format.
//
// "Comment OFFLOAD_BUNDLER_MAGIC_STR__START__ triple"
// Bundle 1
// "Comment OFFLOAD_BUNDLER_MAGIC_STR__END__ triple"
// ...
// "Comment OFFLOAD_BUNDLER_MAGIC_STR__START__ triple"
// Bundle N
// "Comment OFFLOAD_BUNDLER_MAGIC_STR__END__ triple"
class TextFileHandler : public FileHandler {
  /// \brief String that begins a line comment.
  StringRef Comment;

  /// \brief String that initiates a bundle.
  std::string BundleStartString;

  /// \brief String that closes a bundle.
  std::string BundleEndString;

  /// \brief Number of chars read from input.
  size_t ReadChars;

protected:
  void ReadHeader(MemoryBuffer &Input) {}
  StringRef ReadBundleStart(MemoryBuffer &Input) {
    StringRef FC = Input.getBuffer();

    // Find start of the bundle.
    ReadChars = FC.find(BundleStartString, ReadChars);
    if (ReadChars == FC.npos)
      return StringRef();

    // Get position of the triple.
    size_t TripleStart = ReadChars = ReadChars + BundleStartString.size();

    // Get position that closes the triple.
    size_t TripleEnd = ReadChars = FC.find("\n", ReadChars);
    if (TripleEnd == FC.npos)
      return StringRef();

    // Next time we read after the new line.
    ++ReadChars;

    return StringRef(&FC.data()[TripleStart], TripleEnd - TripleStart);
  }
  void ReadBundleEnd(MemoryBuffer &Input) {
    StringRef FC = Input.getBuffer();

    // Read up to the next new line.
    assert(FC[ReadChars] == '\n' && "The bundle should end with a new line.");

    size_t TripleEnd = ReadChars = FC.find("\n", ReadChars + 1);
    if (TripleEnd == FC.npos)
      return;

    // Next time we read after the new line.
    ++ReadChars;

    return;
  }
  void ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) {
    StringRef FC = Input.getBuffer();
    size_t BundleStart = ReadChars;

    // Find end of the bundle.
    size_t BundleEnd = ReadChars = FC.find(BundleEndString, ReadChars);

    StringRef Bundle(&FC.data()[BundleStart], BundleEnd - BundleStart);
    OS << Bundle;
  }

  void WriteHeader(raw_fd_ostream &OS,
                   ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs) {}
  void WriteBundleStart(raw_fd_ostream &OS, StringRef TargetTriple) {
    OS << BundleStartString << TargetTriple << "\n";
    return;
  }
  void WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) {
    OS << BundleEndString << TargetTriple << "\n";
    return;
  }
  void WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) {
    ;
    OS << Input.getBuffer();
    return;
  }

public:
  TextFileHandler(StringRef Comment)
      : FileHandler(), Comment(Comment), ReadChars(0) {
    BundleStartString =
        "\n" + Comment.str() + " " OFFLOAD_BUNDLER_MAGIC_STR "__START__ ";
    BundleEndString =
        "\n" + Comment.str() + " " OFFLOAD_BUNDLER_MAGIC_STR "__END__ ";
  }
};

static void PrintVersion() {
  raw_ostream &OS = outs();
  OS << clang::getClangToolFullVersion("clang-offload-bundler") << '\n';
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::HideUnrelatedOptions(ClangOffloadBundlerCategory);
  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to bundle several input files of the specified type <type> \n"
      "referring to the same source file but different targets into a single \n"
      "one. The resulting file can also be unbundled into different files by \n"
      "this tool if -unbundle is provided.\n");

  if (Help)
    cl::PrintHelpMessage();

  bool Error = false;
  if (Unbundle) {
    if (InputFileNames.size() != 1) {
      Error = true;
      llvm::errs()
          << "error: only one input file supported in unbundling mode.\n";
    }
    if (OutputFileNames.size() != TargetNames.size()) {
      Error = true;
      llvm::errs() << "error: number of output files and targets should match "
                      "in unbundling mode.\n";
    }
  } else {
    if (OutputFileNames.size() != 1) {
      Error = true;
      llvm::errs()
          << "error: only one output file supported in bundling mode.\n";
    }
    if (InputFileNames.size() != TargetNames.size()) {
      Error = true;
      llvm::errs() << "error: number of input files and targets should match "
                      "in bundling mode.\n";
    }
  }

  std::unique_ptr<FileHandler> FH;
  FH.reset(StringSwitch<FileHandler *>(FilesType)
               .Case("i", new TextFileHandler(/*Comment=*/"//"))
               .Case("ii", new TextFileHandler(/*Comment=*/"//"))
               .Case("ll", new TextFileHandler(/*Comment=*/";"))
               .Case("bc", new BinaryFileHandler())
               .Case("s", new TextFileHandler(/*Comment=*/"#"))
               .Case("o", new BinaryFileHandler())
               .Case("gch", new BinaryFileHandler())
               .Case("ast", new BinaryFileHandler())
               .Default(nullptr));

  if (!FH.get()) {
    Error = true;
    llvm::errs() << "error: invalid file type specified.\n";
  }

  if (Error)
    return 1;

  if (Unbundle)
    return FH->Unbundle();
  else
    return FH->Bundle();

  return 0;
}
