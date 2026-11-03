import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class ColabSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "notebooks/workshop_solution.ipynb")
            .read_text(encoding="utf-8")
        )
        cls.source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["id"] == "colab-runtime-bootstrap"
        )

    def setUp(self):
        self.output = io.StringIO()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(self.output))
        self.stack.enter_context(patch.dict(os.environ))
        self.stack.enter_context(patch.object(sys, "path", list(sys.path)))
        self.chdir = self.stack.enter_context(patch("os.chdir"))
        self.run = self.stack.enter_context(patch("subprocess.run"))
        self.invalidate = self.stack.enter_context(patch("importlib.invalidate_caches"))

    def enter_colab(self, *, exists=False, origin=None, head="revision", requested="revision"):
        colab = ModuleType("google.colab")
        self.secret = Mock(return_value="private-test-placeholder")
        colab.userdata = SimpleNamespace(get=self.secret)
        self.stack.enter_context(patch.dict(sys.modules, {"google.colab": colab}))
        self.stack.enter_context(patch("pathlib.Path.exists", return_value=exists))

        def result(command, **kwargs):
            if "get-url" in command:
                stdout = origin or "https://github.com/nutansahoo/aerolex-production-rag.git"
            elif "rev-parse" in command:
                stdout = head if command[-1] == "HEAD" else requested
            else:
                stdout = ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout + "\n")

        self.run.side_effect = result

    def execute(self):
        namespace = {}
        exec(compile(self.source, "notebook:colab-runtime-bootstrap", "exec"), namespace)
        return namespace

    def test_local_runtime_has_no_setup_side_effects(self):
        self.stack.enter_context(patch.dict(sys.modules))
        sys.modules.pop("google.colab", None)
        before_environment = dict(os.environ)
        before_path = list(sys.path)
        self.execute()
        self.run.assert_not_called()
        self.chdir.assert_not_called()
        self.invalidate.assert_not_called()
        self.assertEqual(dict(os.environ), before_environment)
        self.assertEqual(sys.path, before_path)
        self.assertIn("Local runtime", self.output.getvalue())

    def test_fresh_runtime_clones_installs_and_loads_secret(self):
        self.enter_colab()
        namespace = self.execute()
        commands = [call.args[0] for call in self.run.call_args_list]
        self.assertEqual(commands[0][:3], ["git", "clone", "--quiet"])
        self.assertEqual(commands[1][-4:], ["checkout", "--quiet", "--detach", "main"])
        self.assertEqual(commands[2], [sys.executable, "-m", "pip", "install", "-e", "."])
        self.assertTrue(all(call.kwargs["check"] for call in self.run.call_args_list))
        self.assertEqual(str(self.chdir.call_args.args[0]), "/content/aerolex-production-rag")
        self.secret.assert_called_once_with("GEMINI_API_KEY")
        self.assertEqual(os.environ["GEMINI_API_KEY"], "private-test-placeholder")
        self.assertNotIn("private-test-placeholder", self.output.getvalue())
        self.assertNotIn("colab_key", namespace)
        self.assertEqual(sys.path[0], "/content/aerolex-production-rag")

    def test_matching_checkout_is_reused_without_git_mutation(self):
        self.enter_colab(exists=True)
        self.execute()
        commands = [call.args[0] for call in self.run.call_args_list]
        self.assertFalse(any("clone" in command or "checkout" in command
                             or "reset" in command or "pull" in command
                             for command in commands))
        self.assertEqual(commands[-1], [sys.executable, "-m", "pip", "install", "-e", "."])

    def test_wrong_repository_is_rejected_before_install_or_secret_access(self):
        self.enter_colab(exists=True, origin="https://github.com/example/other.git")
        with self.assertRaisesRegex(RuntimeError, "not the expected repository"):
            self.execute()
        self.assertEqual(self.run.call_count, 1)
        self.chdir.assert_not_called()
        self.secret.assert_not_called()

    def test_wrong_revision_is_rejected_without_resetting_work(self):
        self.enter_colab(exists=True, head="other", requested="wanted")
        with self.assertRaisesRegex(RuntimeError, "does not match COLAB_REF"):
            self.execute()
        self.assertEqual(self.run.call_count, 3)
        self.chdir.assert_not_called()
        self.secret.assert_not_called()

    def test_install_failure_stops_before_secret_loading(self):
        self.enter_colab()
        original_side_effect = self.run.side_effect

        def fail_install(command, **kwargs):
            if command[:3] == [sys.executable, "-m", "pip"]:
                raise subprocess.CalledProcessError(1, command)
            return original_side_effect(command, **kwargs)

        self.run.side_effect = fail_install
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        self.secret.assert_not_called()
        self.assertNotIn("Colab setup ready", self.output.getvalue())

    def test_empty_secret_is_actionable(self):
        self.enter_colab()
        self.secret.return_value = ""
        with self.assertRaisesRegex(RuntimeError, "Colab Secrets"):
            self.execute()
        self.assertNotIn("Colab setup ready", self.output.getvalue())

    def test_secret_access_failure_is_not_silently_ignored(self):
        self.enter_colab()
        self.secret.side_effect = PermissionError("Enable notebook access")
        with self.assertRaisesRegex(PermissionError, "notebook access"):
            self.execute()
        self.assertNotIn("Colab setup ready", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
