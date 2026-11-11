#!/usr/bin/env python3
"""Keep generated results local and automatically untrack files >= 50 MiB.

Only the Git index, ignore rules and local hook are modified. Research data
and results in the working directory are never removed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

LIMIT = 50 * 1024 * 1024
MARKER = "astrodynamics-lightweight-v1"
START = "# BEGIN " + MARKER
END = "# END " + MARKER
OUTPUT_DIRS = (
    "CR3BP/Lunar Swingby/Clustering/Clusters",
    "CR3BP/Ballistic Captures/Clustering/Clusters",
    "CR3BP/Lunar Swingby/Heliocentric/GA18_real_asteroids",
    "CR3BP/Lunar Swingby/Heliocentric/GA18_parking_orbits",
    "CR3BP/Lunar Swingby/Heliocentric/GA18_parking_plots",
    "CR3BP/Lunar Swingby/Heliocentric/GA18_parking_plots_reali_planari",
)
SOURCE_SUFFIXES = (
    ".py", ".ipynb", ".m", ".jl", ".sh", ".c", ".h", ".cpp",
    ".hpp", ".f", ".f90", ".tex", ".md", ".rst", ".toml",
    ".yaml", ".yml",
)


def git(*args, data=None, check=True, literal=False):
    env = os.environ.copy()
    if literal:
        env["GIT_LITERAL_PATHSPECS"] = "1"
    result = subprocess.run(["git", *args], input=data, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    if check and result.returncode:
        raise RuntimeError(os.fsdecode(result.stderr).strip())
    return result


def git_path(name):
    return Path(os.fsdecode(git("rev-parse", "--git-path", name).stdout).strip()).resolve()


def output_path(path):
    return (any(path.startswith(root + "/") for root in OUTPUT_DIRS)
            and not path.lower().endswith(SOURCE_SUFFIXES))


def generated_path(path):
    return output_path(path) or "__pycache__" in path.split("/") or path.endswith((".pyc", ".pyo"))


def index_entries():
    entries = {}
    for record in git("ls-files", "--stage", "-z").stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0":
            raise RuntimeError("Resolve index conflicts before running cleanup.")
        entries[os.fsdecode(raw_path)] = (mode.decode(), oid.decode())
    return entries


def object_sizes(entries):
    oids = sorted({oid for mode, oid in entries.values() if mode != "160000"})
    if not oids:
        return {}
    result = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                 data=("\n".join(oids) + "\n").encode())
    sizes = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[1] != b"blob":
            raise RuntimeError("Cannot inspect every staged object; cleanup stopped.")
        sizes[parts[0].decode()] = int(parts[2])
    return sizes


def large_workfile(path, mode="100644"):
    if mode in ("120000", "160000"):
        return False
    try:
        info = Path(path).lstat()
        return stat.S_ISREG(info.st_mode) and info.st_size >= LIMIT
    except FileNotFoundError:
        return False


def select(entries):
    sizes = object_sizes(entries)
    selected = {}
    for path, (mode, oid) in entries.items():
        size = sizes.get(oid, 0)
        if generated_path(path) or size >= LIMIT or large_workfile(path, mode):
            selected[path] = size
    return selected


def literal_ignore(path):
    if "\n" in path or "\r" in path:
        raise RuntimeError("A large filename contains a newline; review it manually.")
    return "/" + "".join("\\" + char if char in "\\[]*?#! " else char for char in path)


def ignore_block(previous, extra):
    retained = []
    outside = previous
    if START in previous:
        before, block = previous.split(START, 1)
        if END not in block:
            raise RuntimeError("Incomplete managed ignore block; no changes made.")
        old, after = block.split(END, 1)
        retained = [line for line in old.splitlines() if line.startswith("/")]
        outside = before.rstrip("\n") + after
    lines = [START, "# Generated output stays on this computer.", "**/__pycache__/", "*.py[cod]"]
    for root in OUTPUT_DIRS:
        lines.extend(["/" + root + "/**", "!/" + root + "/**/"])
        lines.extend("!/" + root + "/**/*" + ext for ext in SOURCE_SUFFIXES)
    # Retain prior exact exclusions when a large file is not part of an output directory.
    generated_rules = {"/" + root + "/**" for root in OUTPUT_DIRS}
    exact = set(retained) - generated_rules
    exact.update(literal_ignore(path) for path in extra if not generated_path(path))
    lines.extend(sorted(exact))
    lines.extend([END, ""])
    return outside.rstrip("\n") + "\n\n" + "\n".join(lines)


def update_ignore(extra):
    path = Path(".gitignore")
    if path.is_symlink():
        raise RuntimeError("Refusing to replace a symlinked .gitignore.")
    staged = git("show", ":.gitignore", check=False)
    staged_text = os.fsdecode(staged.stdout) if staged.returncode == 0 else ""
    working_text = path.read_text(encoding="utf-8", errors="surrogateescape") if path.exists() else staged_text
    staged_new = os.fsencode(ignore_block(staged_text, extra))
    working_new = os.fsencode(ignore_block(working_text, extra))
    if path.exists() and path.read_bytes() == working_new and staged.stdout == staged_new:
        return
    path.write_bytes(working_new)
    oid = git("hash-object", "-w", "--stdin", data=staged_new).stdout.decode().strip()
    git("update-index", "--add", "--cacheinfo", "100644," + oid + ",.gitignore")


def untrack(selected):
    if not selected:
        return
    before = {}
    for path in selected:
        try:
            info = Path(path).lstat()
            before[path] = (info.st_size, info.st_mtime_ns, info.st_mode)
        except FileNotFoundError:
            pass
    git("rm", "--cached", "-f", "--quiet", "--ignore-unmatch",
        "--pathspec-from-file=-", "--pathspec-file-nul",
        data=b"\0".join(os.fsencode(path) for path in selected) + b"\0", literal=True)
    for path, expected in before.items():
        info = Path(path).lstat()
        if (info.st_size, info.st_mtime_ns, info.st_mode) != expected:
            raise RuntimeError("A working file changed concurrently: " + path)


def report(entries, selected):
    print(json.dumps({"tracked_files": len(entries), "untrack_files": len(selected),
                      "untrack_MiB": round(sum(selected.values()) / 1048576, 2),
                      "largest": sorted(selected.items(), key=lambda item: -item[1])[:15]}, indent=2))


def hook_content():
    return """#!/bin/sh
# astrodynamics-lightweight-v1
repo_root=$(git rev-parse --show-toplevel) || exit 1
python3 "$repo_root/tools/git_lightweight.py" pre-commit || exit $?
old_hook=$(git rev-parse --git-path hooks/pre-commit.before-size-guard) || exit 1
if [ -x "$old_hook" ]; then
    "$old_hook" "$@" || exit $?
fi
"""


def setup(entries):
    if git("config", "--get", "core.hooksPath", check=False).returncode == 0:
        raise RuntimeError("A custom core.hooksPath is configured. Integrate the size guard with that hook first.")
    for name in ("index.lock", "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply", "sequencer"):
        if git_path(name).exists():
            raise RuntimeError("Another Git operation is active: " + name)
    origin = git("remote", "get-url", "origin").stdout.decode().strip().rstrip("/")
    if not origin.removesuffix(".git").endswith(("github.com/lucacecca01/Astrodynamics", "github.com:lucacecca01/Astrodynamics")):
        raise RuntimeError("Unexpected origin. Run setup only in lucacecca01/Astrodynamics.")
    selected = select(entries)
    extra = set(selected)
    for raw in git("ls-files", "--others", "--exclude-standard", "-z").stdout.split(b"\0"):
        if raw:
            path = os.fsdecode(raw)
            if not generated_path(path) and large_workfile(path):
                extra.add(path)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    backup = git_path("cleanup-backups/" + stamp)
    backup.mkdir(parents=True)
    script = Path("tools/git_lightweight.py")
    source = Path(__file__).read_bytes()
    if script.exists() and script.read_bytes() != source:
        raise RuntimeError("tools/git_lightweight.py already has different contents; refusing to overwrite it.")
    hook = git_path("hooks/pre-commit")
    chain = git_path("hooks/pre-commit.before-size-guard")
    if hook.is_symlink() or chain.is_symlink():
        raise RuntimeError("Refusing to replace symlinked Git hooks.")
    if hook.exists() and MARKER not in hook.read_text(errors="replace") and chain.exists():
        raise RuntimeError("A previous hook backup already exists; refusing to overwrite it.")
    # Save the index, configuration and hook bytes before changing any of them.
    paths = [git_path("index"), Path(".gitignore"), script, hook, chain]
    saved = {}
    for number, path in enumerate(paths):
        if path.exists():
            saved[path] = (path.read_bytes(), path.stat().st_mode)
            shutil.copy2(path, backup / (str(number) + "-" + path.name))
        else:
            saved[path] = None
    git("branch", "backup-before-output-cleanup-" + stamp, "HEAD")
    try:
        update_ignore(extra)
        untrack(selected)
        script.parent.mkdir(exist_ok=True)
        script.write_bytes(source)
        git("add", "--", str(script), literal=True)
        hook.parent.mkdir(parents=True, exist_ok=True)
        if hook.exists() and MARKER not in hook.read_text(errors="replace"):
            shutil.copy2(hook, chain)
        hook.write_text(hook_content())
        hook.chmod(0o755)
        remaining = select(index_entries())
        if remaining:
            raise RuntimeError("Some generated/large files are still staged.")
    except Exception:
        for path, original in saved.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original[0])
                path.chmod(original[1])
        raise
    report(entries, selected)
    print("Working files preserved. Index and hook backup: " + str(backup))
    print("Cleanup is staged. Review, Commit and Sync in VS Code.")
    print("Earlier commits remain in history; this operation does not rewrite history.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("report", "setup", "pre-commit"), default="report", nargs="?")
    args = parser.parse_args()
    root = git("rev-parse", "--show-toplevel").stdout
    os.chdir(os.fsdecode(root).strip())
    entries = index_entries()
    if args.action == "setup":
        setup(entries)
        return
    if args.action == "pre-commit":
        names = {os.fsdecode(path) for path in git("diff", "--cached", "--no-renames", "--name-only", "--diff-filter=AMT", "-z").stdout.split(b"\0") if path}
        entries = {path: value for path, value in entries.items() if path in names}
    selected = select(entries)
    if args.action == "report":
        report(entries, selected)
    elif selected:
        update_ignore(selected)
        untrack(selected)
        print("Kept locally, excluded from Git: " + str(len(selected)) + " generated/large file(s).")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print("Cleanup stopped: " + str(exc), file=sys.stderr)
        sys.exit(1)
