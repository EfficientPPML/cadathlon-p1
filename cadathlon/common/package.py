"""Local archives only; EXCLUDE applies to contestant releases."""
import argparse
import ast
import fnmatch
from pathlib import Path
import shutil
import sys
import tarfile


COMMON_RUNTIME_FILES = (
    "support.py", "Makefile.problem", "package.py", "transformer_runner.py",
    "encrypted_polynomial.py", "encrypted_transformer.py", "encrypted_reference.py",
    "encrypted_refresh.py", "encrypted_rotation.py", "encrypted_compilation.py",
    "encrypted_primitives.py", "encrypted_ntt.py", "encrypted_packed_qwen.py", "encrypted_packed_transformer.py",
    "encrypted_bounded_arithmetic.py", "encrypted_bounded_transformer.py",
    "encrypted_streaming_linear.py", "encrypted_chained_streaming_linear.py",
    "encrypted_integer_affine.py", "encrypted_scaled_newton.py",
)
PUBLIC_RELEASE_EVIDENCE = ("validation/p1_student_suite.json",
                           "validation/p1_student_v6e8/report.json",
                           "validation/p1_legacy_suite.json",
                           "validation/p3_compact_baseline.json")
# Submissions that need only part of the shared runtime: P1 imports just the
# security gate in support.py, and P5 the WISE subset below. Every other
# problem's submission carries all of COMMON_RUNTIME_FILES, and the contestant
# release always does. `submit` adds any further shared module a submission's
# sources import (import_closure).
SHARED_TOOLING_FILES = ("support.py", "Makefile.problem", "package.py")
WISE_RUNTIME_FILES = (
    "encrypted_polynomial.py", "encrypted_compilation.py",
    "encrypted_primitives.py", "encrypted_rotation.py",
    "encrypted_ntt.py", "encrypted_refresh.py",
)
SUBMISSION_RUNTIME = {
    "p1_benchmark": (SHARED_TOOLING_FILES, "declared archive dependency"),
    "p5_wise": (SHARED_TOOLING_FILES + WISE_RUNTIME_FILES, "WISE runtime dependency"),
}


def submission_sources(src):
    """The Python sources `submit` ships from src/: regular files outside run/ and results/."""
    return [path for path in src.rglob("*.py")
            if path.is_file() and not path.is_symlink()
            and not {"run", "results"}.intersection(path.relative_to(src).parts)]


def shared_imports(path, common):
    """The common/ modules that one Python source imports by name."""
    found = set()
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules = [node.module]
        else:
            continue
        for module in modules:
            name = module.split(".")[0] + ".py"
            # A linked module, even a broken link, is kept so that required_files
            # rejects it instead of the archive silently lacking it.
            if (common / name).is_file() or (common / name).is_symlink():
                found.add(name)
    return found


def import_closure(sources, common):
    """Every common/ module the sources import, directly or through each other.

    A source that does not parse (or exceeds the parser's limits) cannot be
    imported either, so it is still
    archived but its imports are not followed (with a warning).
    """
    parsed = {}

    def imports_of(path):
        # Each file is parsed once, sources and shared modules alike.
        if path not in parsed:
            try:
                parsed[path] = shared_imports(path, common)
            except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
                print(f"warning: {path} does not parse ({error}); its imports are not followed",
                      file=sys.stderr)
                parsed[path] = set()
        return parsed[path]

    pending, seen = set(), set()
    for path in sources:
        pending |= imports_of(path)
    while pending:
        name = pending.pop()
        seen.add(name)
        module = common / name
        if module.is_file() and not module.is_symlink():
            pending |= imports_of(module) - seen
    return seen


def required_files(root, names, kind="declared archive dependency"):
    files = [root / name for name in names]
    missing = [str(path) for path in files if not path.is_file() or path.is_symlink()]
    if missing:
        raise FileNotFoundError(f"Missing {kind}: " + ", ".join(missing))
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("clean", "submit", "release"))
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    directory = args.directory.resolve()
    if args.action == "clean":
        for path in directory.rglob("__pycache__"):
            shutil.rmtree(path)
        # Sockets, pidfiles and daemon logs from a suite run; scratch, not source.
        for path in directory.rglob("suite/run"):
            if path.is_dir():
                shutil.rmtree(path)
        (directory / "submission.tar.gz").unlink(missing_ok=True)
        return
    if args.action == "submit":
        for required in ("REPORT.md", "cross.patch"):
            if not (directory / required).is_file():
                raise FileNotFoundError(f"Create {directory / required} before submission")
        # Preserve the non_agent_pset/agent_pset level: runners resolve common
        # three parents above src, just as they do in the source checkout.
        root = directory.parents[1]
        candidates = [directory / "REPORT.md", directory / "cross.patch"]
        sources = submission_sources(directory / "src")
        candidates += sources
        candidates += [directory / "src" / "Makefile"]
        candidates += [directory / "src" / "suite" / "suite_config.json"]
        candidates += list(directory.glob("results*.json"))
        runtime, kind = SUBMISSION_RUNTIME.get(
            directory.name, (COMMON_RUNTIME_FILES, "declared archive dependency"))
        # The shared tooling is reported as such; only the problem's own subset
        # carries its label.
        shipped = required_files(root / "common",
                                 [name for name in runtime if name in SHARED_TOOLING_FILES])
        shipped += required_files(root / "common",
                                  [name for name in runtime if name not in SHARED_TOOLING_FILES], kind)
        # The declared runtime is a floor: also ship every other shared module
        # the sources (or the shipped modules) import by an import statement, so
        # a new static import in a contestant's code still reaches the archive.
        # Modules loaded dynamically (importlib, support.load_module) must be in
        # the declared runtime.
        python = sources + [path for path in shipped if path.suffix == ".py"]
        shipped += required_files(root / "common", sorted(import_closure(python, root / "common")),
                                  "imported shared module")
        candidates += shipped
        output = directory / "submission.tar.gz"
    else:
        root = directory
        output = root / "contestant-release.tar.gz"
        candidates = []
        for problem in ("non_agent_pset/p1_benchmark", "agent_pset/p2_bert",
                        "non_agent_pset/p3_cnn", "agent_pset/p4_qwen",
                        "agent_pset/p5_wise"):
            folder = root / problem
            patterns = [line.strip()
                        for line in (folder / "EXCLUDE").read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#")]
            for path in folder.rglob("*"):
                rel = path.relative_to(folder).as_posix()
                if not any(fnmatch.fnmatchcase(rel, pattern) for pattern in patterns):
                    candidates.append(path)
        candidates += required_files(root / "common", COMMON_RUNTIME_FILES)
        # Only the public aggregate reports cited by student documentation.
        # Per-image organizer verification evidence remains excluded above.
        candidates += required_files(root, PUBLIC_RELEASE_EVIDENCE)
    files = sorted({p for p in candidates if p.is_file() and not p.is_symlink()
                    and "__pycache__" not in p.parts and p.suffix not in (".gz", ".golden", ".pyc")})
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(root), recursive=False)
    print(f"Created {output} ({len(files)} files); no upload performed")


if __name__ == "__main__":
    main()
