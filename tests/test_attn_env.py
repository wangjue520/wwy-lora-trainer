# -*- coding: utf-8 -*-
"""「注意力实现」选了 xformers，训练环境里却没装（群友报：Krea 2 训练报
AttributeError: 'NoneType' object has no attribute 'memory_efficient_attention'）。

musubi-tuner 导入 xformers 失败时不报错（xops = None），等真正算注意力才炸，训练器也认不出这个报错。
修复：开训前先检查，xformers 没有就自动装（钉住 PyTorch）；装不了的（flash-attn / SageAttention、安装失败）拦下来，
给「重新安装」/「改回默认注意力」；训练中再报这个错也能认出来。

运行：python -m unittest discover -s tests"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import diagnose  # noqa: E402
import envsetup  # noqa: E402
import models  # noqa: E402
import tasks  # noqa: E402

TAIL = """  File "D:\\WWY-LoRA-Trainer\\engines\\musubi-tuner\\src\\musubi_tuner\\modules\\attention.py", line 219, in attention
    x_i = xops.memory_efficient_attention(qi, ki, vi, p=drop_rate)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'memory_efficient_attention'
===== 失败：训练意外中止 ====="""


class EnsureAttn(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.patches = [mock.patch.object(tasks, "_set", side_effect=lambda t, **kw: t.update(kw))]
        for p in self.patches:
            p.start()
        tasks._state["stop_req"] = False

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def run_ensure(self, model, mode, check, install=None):
        t = {"model_type": model, "params": {"attn_mode": mode}}
        calls = []

        def fake_install(job, m):
            calls.append(m)
            if install:
                raise RuntimeError(install)
            return True
        with mock.patch.object(envsetup, "attn_check", side_effect=lambda m: check.pop(0) if check else None) as chk, \
                mock.patch.object(envsetup, "install_attn", side_effect=fake_install):
            ok = tasks._ensure_attn(t, models.get(model), self.lines.append)
        return ok, t, calls, chk

    def test_default_sdpa_skips_check(self):
        ok, t, calls, chk = self.run_ensure("krea2", "sdpa", [])
        self.assertTrue(ok)
        chk.assert_not_called()
        self.assertEqual(calls, [])

    def test_xformers_present(self):
        ok, t, calls, _ = self.run_ensure("krea2", "xformers", [None])
        self.assertTrue(ok)
        self.assertEqual(calls, [])

    def test_xformers_missing_is_installed(self):
        ok, t, calls, _ = self.run_ensure("krea2", "xformers", ["ModuleNotFoundError: No module named 'xformers'"])
        self.assertTrue(ok)
        self.assertEqual(calls, ["xformers"])
        self.assertTrue(any("装好了" in ln for ln in self.lines))
        self.assertNotEqual(t.get("status"), "blocked")

    def test_install_failure_blocks_with_fixes(self):
        ok, t, calls, _ = self.run_ensure("anima", "xformers", ["No module named 'xformers'"], install="pip 安装失败")
        self.assertFalse(ok)
        self.assertEqual(t["status"], "blocked")
        fixes = t["issues"][0]["fixes"]
        self.assertEqual([f["id"] for f in fixes], ["install_attn", "set_params"])
        self.assertEqual(fixes[1]["data"]["values"], {"attn_mode": "torch"})   # Anima 系默认写 torch
        self.assertIn("pip 安装失败", t["issues"][0]["reason"])

    def test_flash_cannot_be_installed(self):
        ok, t, calls, _ = self.run_ensure("krea2", "flash_attn", ["No module named 'flash_attn'"])
        self.assertFalse(ok)
        self.assertEqual(calls, [])
        fixes = t["issues"][0]["fixes"]
        self.assertEqual([f["id"] for f in fixes], ["set_params"])
        self.assertEqual(fixes[0]["data"]["values"], {"attn_mode": "sdpa"})   # Krea 2 默认写 sdpa


class Diagnose(unittest.TestCase):
    def test_nonetype_xops_is_recognized(self):
        d = diagnose.diagnose(TAIL, {"model_type": "krea2", "params": {"attn_mode": "xformers"}})
        self.assertEqual(d["code"], "xformers")
        self.assertEqual([f["id"] for f in d["fixes"]], ["install_attn", "set_params"])
        self.assertEqual(d["fixes"][1]["data"]["values"], {"attn_mode": "sdpa"})


class CheckOutput(unittest.TestCase):
    def test_color_codes_stripped(self):
        class R:
            returncode = 1
            stderr = b"Traceback...\n\x1b[1;35mModuleNotFoundError\x1b[0m: \x1b[35mNo module named 'xformers'\x1b[0m\n"
        with mock.patch.object(envsetup.subprocess, "run", return_value=R()), \
                mock.patch.object(envsetup.paths, "env_python", return_value="python"):
            self.assertEqual(envsetup.attn_check("xformers"), "ModuleNotFoundError: No module named 'xformers'")

    def test_unknown_mode_needs_nothing(self):
        self.assertIsNone(envsetup.attn_check("sdpa"))
        self.assertIsNone(envsetup.attn_check("torch"))


if __name__ == "__main__":
    unittest.main()
