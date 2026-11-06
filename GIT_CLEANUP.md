# Local results and lightweight Git commits

The six generated-output directories listed in `tools/git_lightweight.py` stay
on the research computer. Python, notebooks, MATLAB, Julia and other source
files remain eligible for tracking, including source files placed under those
directories. Other files of 50 MiB or more are excluded by the installed guard.

## Apply safely to an existing research checkout

Wait for any running batch-upload script to finish. Do not merge the cleanup
branch and pull its output deletions into the research checkout first: a normal
pull could remove previously tracked results from disk.

From `/home/lucacecca/Astrodynamics`, run the setup script from a temporary copy:

```sh
git fetch origin cleanup/local-results-20261105
git show origin/cleanup/local-results-20261105:tools/git_lightweight.py > /tmp/astrodynamics-git-cleanup.py
python3 /tmp/astrodynamics-git-cleanup.py setup
```

Setup saves a backup branch, the original index, ignore rules and existing hook
under `.git/cleanup-backups/`. It uses `git rm --cached` and checks that working
files retain their size, modification time and mode. It stages cleanup and the
helper without committing or pushing. Review the changes, then use VS Code
Commit and Sync normally. The local pre-commit guard automatically excludes new
large/generated additions, preserving their working copies. Existing default
hooks are chained and LFS pre-push hooks remain in place. A custom
`core.hooksPath` causes setup to stop rather than overwrite another hook system.

The draft branch is a reviewable preview. Running local setup and committing on
the current branch is the supported way to preserve existing local results;
the draft PR does not also need to be merged afterward.

## What this cleanup changes

The preview was built from `beeda8bc491578a262ddfc074c8dc49a3d33b27c`
(`Update batch 12`). It excludes 30,048 generated/cache files totaling
22.2507 GiB from the current tree, retaining all source files and about
329.12 MiB of existing tracked content, plus this helper and documentation.
Input datasets outside the generated directories remain eligible below 50 MiB.

Older commits still contain the old outputs. Making historical clones smaller
requires a separate history rewrite and coordinated remote update; neither
setup nor this preview rewrites published history or force-pushes a branch.

## Verification

```sh
python3 -m unittest discover -s tools -p 'test_git_lightweight.py' -v
python3 tools/git_lightweight.py report
```

The integration checks cover data preservation, code inside output directories,
literal filenames, automatic exclusion during a normal commit, preserving
unstaged ignore edits and existing hooks, small LFS-style pointers with large
working files, symlinks, and repeatable installation.
