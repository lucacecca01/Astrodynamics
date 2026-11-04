"""Integration checks for local-data preservation and ordinary GUI commits."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("git_lightweight.py").resolve()
OUTPUT = "CR3BP/Lunar Swingby/Heliocentric/GA18_parking_orbits"
LARGE = 50 * 1024 * 1024 + 1


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Cleanup test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("remote", "add", "origin", "https://github.com/lucacecca01/Astrodynamics.git")
        self.write(".gitignore", "existing.tmp\n")
        self.write("science.py", "print('science')\n")
        self.git("add", ".")
        self.git("commit", "-qm", "initial")

    def git(self, *args, check=True):
        return subprocess.run(["git", *args], cwd=self.root, check=check,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def large(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.truncate(LARGE)
        return path

    def run_setup(self, check=True):
        return subprocess.run([sys.executable, str(SCRIPT), "setup"], cwd=self.root,
                              check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_cleanup_keeps_data_code_staging_and_existing_hooks(self):
        output = self.write(OUTPUT + "/plot.png", "generated image")
        source = self.write(OUTPUT + "/helper.py", "print('keep me')\n")
        heavy = self.large("input with [brackets].bin")
        self.git("add", ".")
        original = (output.read_bytes(), heavy.stat().st_size, source.read_bytes())
        prepush = self.write(".git/hooks/pre-push", "#!/bin/sh\nexit 0\n")
        old_hook = self.write(".git/hooks/pre-commit", "#!/bin/sh\necho ran > .git/previous-hook-ran\n")
        old_hook.chmod(0o755)
        prepush_bytes = prepush.read_bytes()
        self.run_setup()
        self.assertEqual(original, (output.read_bytes(), heavy.stat().st_size, source.read_bytes()))
        tracked = self.git("ls-files", "-z").stdout.split(b"\0")
        self.assertNotIn(os.fsencode(OUTPUT + "/plot.png"), tracked)
        self.assertNotIn(b"input with [brackets].bin", tracked)
        self.assertIn(os.fsencode(OUTPUT + "/helper.py"), tracked)
        self.assertEqual(prepush_bytes, prepush.read_bytes())
        self.git("commit", "-qm", "cleanup")
        self.assertTrue((self.root / ".git/previous-hook-ran").exists())
        self.assertTrue(list((self.root / ".git/cleanup-backups").glob("*/0-index")))
        self.assertEqual(self.git("check-ignore", "--", OUTPUT + "/helper.py", check=False).returncode, 1)
        self.assertEqual(self.git("check-ignore", "--", "input with [brackets].bin").returncode, 0)

    def test_hook_automatically_excludes_new_large_files_without_staging_unrelated_ignore_edits(self):
        self.run_setup()
        self.git("commit", "-qm", "install cleanup")
        ignored = self.root / ".gitignore"
        ignored.write_text(ignored.read_text() + "\nlocal-user-change.tmp\n")
        heavy = self.large("new large [v1].bin")
        self.write("science.py", "print('updated')\n")
        self.git("add", "--", "new large [v1].bin", "science.py")
        self.git("commit", "-qm", "ordinary GUI-style commit")
        self.assertEqual(heavy.stat().st_size, LARGE)
        self.assertNotIn(b"new large [v1].bin", self.git("ls-tree", "-r", "--name-only", "HEAD").stdout.splitlines())
        committed_ignore = self.git("show", "HEAD:.gitignore").stdout
        self.assertNotIn(b"local-user-change.tmp", committed_ignore)
        self.assertIn(b"local-user-change.tmp", self.git("diff", "--", ".gitignore").stdout)
        self.assertEqual(self.git("check-ignore", "--", "new large [v1].bin").returncode, 0)

    def test_large_working_file_with_small_staged_pointer_is_excluded(self):
        path = self.write("lfs-style-data.bin", "small staged pointer\n")
        self.git("add", "--", "lfs-style-data.bin")
        with path.open("wb") as stream:
            stream.truncate(LARGE)
        self.run_setup()
        self.assertEqual(path.stat().st_size, LARGE)
        self.assertNotIn(b"lfs-style-data.bin", self.git("ls-files", "-z").stdout.split(b"\0"))

    def test_report_is_read_only_and_custom_hooks_are_not_overwritten(self):
        self.write(OUTPUT + "/result.txt", "result")
        self.git("add", ".")
        before_index = (self.root / ".git/index").read_bytes()
        before_ignore = (self.root / ".gitignore").read_bytes()
        subprocess.run([sys.executable, str(SCRIPT), "report"], cwd=self.root, check=True, stdout=subprocess.PIPE)
        self.assertEqual(before_index, (self.root / ".git/index").read_bytes())
        self.git("config", "core.hooksPath", ".custom-hooks")
        result = self.run_setup(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"core.hooksPath", result.stderr)
        self.assertEqual(before_index, (self.root / ".git/index").read_bytes())
        self.assertEqual(before_ignore, (self.root / ".gitignore").read_bytes())

    def test_symlink_to_large_file_is_retained_and_setup_is_repeatable(self):
        self.large("outside.bin")
        (self.root / "data-link").symlink_to("outside.bin")
        self.git("add", "--", "data-link")
        self.run_setup()
        first = (self.root / ".gitignore").read_bytes()
        self.run_setup()
        self.assertEqual(first, (self.root / ".gitignore").read_bytes())
        self.assertIn(b"data-link", self.git("ls-files", "-z").stdout.split(b"\0"))


if __name__ == "__main__":
    unittest.main()
