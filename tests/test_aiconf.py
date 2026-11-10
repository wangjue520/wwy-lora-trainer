# -*- coding: utf-8 -*-
"""报错诊断的 AI 设置和 WWY 启动器同步（app/aiconf.py）。共用文件、训练器设置、启动器配置都用临时的。

运行：python -m unittest discover -s tests"""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import aiconf  # noqa: E402
import settings  # noqa: E402


class AiConfSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.shared = os.path.join(self.tmp, "WWY", "ai_api.json")
        self.store = dict(settings.DEFAULTS)
        self.launcher = None   # (values, mtime) 或 None
        lc = os.path.join(self.tmp, "launcher_config.json")
        self.lc_path = lc
        self.patches = [
            mock.patch.dict(os.environ, {"WWY_AI_API_FILE": self.shared}),
            mock.patch.object(settings, "load", side_effect=lambda: dict(self.store)),
            mock.patch.object(settings, "save", side_effect=self._save),
            mock.patch.object(aiconf, "launcher_config", side_effect=self._launcher),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _save(self, patch):
        self.store.update(patch)
        return dict(self.store)

    def _launcher(self):
        if not self.launcher:
            return None
        return aiconf._vals(self.launcher[0]), self.launcher[1], self.lc_path

    def launcher_writes(self, mtime=None, **vals):
        self.launcher = (vals, mtime or time.time())

    def launcher_writes_shared(self, **vals):
        """新版启动器（app/ai_api_sync.py）改了设置：写共用文件。"""
        os.makedirs(os.path.dirname(self.shared), exist_ok=True)
        with open(self.shared, "w", encoding="utf-8") as f:
            json.dump({"values": aiconf._vals(vals), "updated": time.time(), "by": "launcher"}, f)

    def test_old_launcher_key_is_used(self):
        self.launcher_writes(diag_llm_provider="qwen", diag_llm_api_key="sk-launcher")
        v = aiconf.sync()
        self.assertEqual(v["diag_llm_api_key"], "sk-launcher")
        self.assertEqual(aiconf.read_shared()["values"]["diag_llm_api_key"], "sk-launcher")
        self.assertEqual(aiconf.resolve()[3], "sk-launcher")

    def test_trainer_save_goes_to_shared_and_survives_stale_launcher(self):
        self.launcher_writes(diag_llm_api_key="sk-old")
        aiconf.sync()
        aiconf.save({"diag_llm_provider": "deepseek", "diag_llm_api_key": "sk-trainer"})
        self.assertEqual(aiconf.read_shared()["values"]["diag_llm_api_key"], "sk-trainer")
        # 老启动器开着、存了别的设置（配置文件更新了，但诊断设置还是旧的）：不能把训练器刚改的盖回去
        self.launcher_writes(diag_llm_api_key="sk-old")
        self.assertEqual(aiconf.sync()["diag_llm_api_key"], "sk-trainer")

    def test_new_launcher_change_is_pulled(self):
        aiconf.save({"diag_llm_api_key": "sk-a"})
        self.launcher_writes_shared(diag_llm_provider="claude", diag_llm_api_key="sk-ant-b")
        v = aiconf.sync()
        self.assertEqual(v["diag_llm_api_key"], "sk-ant-b")
        self.assertEqual(aiconf.resolve()[0], "anthropic")

    def test_old_launcher_edit_after_sync_is_pulled(self):
        self.launcher_writes(diag_llm_api_key="sk-1")
        aiconf.sync()
        self.launcher_writes(diag_llm_api_key="sk-2")   # 用户在老版启动器里换了 Key
        self.assertEqual(aiconf.sync()["diag_llm_api_key"], "sk-2")

    def test_launcher_without_key_does_not_clear(self):
        aiconf.save({"diag_llm_api_key": "sk-mine"})
        self.launcher_writes(diag_llm_provider="deepseek", diag_llm_api_key="")
        self.assertEqual(aiconf.sync()["diag_llm_api_key"], "sk-mine")

    def test_nothing_configured(self):
        v = aiconf.sync()
        self.assertEqual(v["diag_llm_api_key"], "")
        self.assertIsNone(aiconf.resolve())
        self.assertFalse(os.path.exists(self.shared))

    def test_public_hides_key(self):
        aiconf.save({"diag_llm_api_key": "sk-abcdefghijklmn"})
        p = aiconf.public()
        self.assertTrue(p["has_key"])
        self.assertNotIn("sk-abcdefghijklmn", json.dumps(p))


if __name__ == "__main__":
    unittest.main()
