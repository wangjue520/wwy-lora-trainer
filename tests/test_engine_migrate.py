# -*- coding: utf-8 -*-
"""Windows 上的 MOSS 专用引擎任务（群友报：「训练器内部错误：MOSS 专用引擎只支持 Linux」）。

原来：专用引擎只支持 Linux，Windows 界面虽然不列出了，但升级前排进队列的 / 从 Linux 存档带过来的任务还是专用引擎，
开训就报错；报错让人去「模型设置」切回 sd-scripts，可 Windows 上根本没有引擎开关，「重试」也只会原样再失败。
修复：队列读取 / 重试 / 开训前 / 加入队列时，把本机跑不了的引擎换成同一底模能用的引擎（sd-scripts），
两个引擎都有的参数和模型带过去，并按显存套用推荐档（显存小时自动开 int8 量化）。

运行：python -m unittest discover -s tests"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import models  # noqa: E402
import plat  # noqa: E402
import settings  # noqa: E402
import tasks  # noqa: E402

NATIVE_MODELS = {"dit": r"X:\m\moss_image_base_bf16.safetensors", "te": r"X:\m\qwen3.5_9b_bf16.safetensors",
                 "vae": r"X:\m\vae\diffusion_pytorch_model.safetensors"}


def native_params(**kw):
    p = dict(models.get("moss_native").defaults())
    p.update(dataset_dir=r"X:\data\huaf", output_name="mylora_moss_v1_huaf", learning_rate=1e-4,
             network_type="lokr", checkpoint_policy="attention", config_dir=r"X:\cfg")
    p.update(kw)
    return p


class Convert(unittest.TestCase):
    def test_windows_switches_to_sd_scripts_and_keeps_shared_params(self):
        with mock.patch.object(plat, "IS_WIN", True):
            dst, p, m, note = models.convert("moss_native", native_params(), NATIVE_MODELS)
        self.assertEqual(dst.id, "moss")
        self.assertEqual(p["dataset_dir"], r"X:\data\huaf")
        self.assertEqual(p["output_name"], "mylora_moss_v1_huaf")
        self.assertEqual(p["learning_rate"], 1e-4)            # 自己改过的、两边都有的参数带过去
        self.assertEqual(p["network_type"], "lokr")           # sd-scripts 版也有 LoKr
        self.assertNotIn("checkpoint_policy", p)              # 专用引擎才有的参数不带
        self.assertNotIn("config_dir", p)
        self.assertEqual(p["timestep_sampling"], "sigma")     # 没改过的用 sd-scripts 自己的默认值
        self.assertEqual(m, NATIVE_MODELS)
        self.assertIn("sd-scripts", note)

    def test_small_vram_uses_int8_quant_preset(self):
        with mock.patch.object(plat, "IS_WIN", True):
            _, p, _, note = models.convert("moss_native", native_params(), NATIVE_MODELS, vram_gb=12)
        self.assertTrue(p["convrot_int8"])
        self.assertTrue(p["convrot_int8_text_encoder"])
        self.assertEqual(p["blocks_to_swap"], 24)
        self.assertIn("12 GB", note)

    def test_quantized_model_files_carry_over(self):
        q = dict(NATIVE_MODELS, dit=r"X:\m\moss_image_sft_int8_convrot.safetensors",
                 te=r"X:\m\qwen3.5_9b_int8_convrot.safetensors")
        with mock.patch.object(plat, "IS_WIN", True):
            dst, _, m, _ = models.convert("moss_native", native_params(), q)
        self.assertEqual(m, q)
        self.assertTrue(all(dst.accept(s, {"name": os.path.basename(q[s.key]), "kind": s.kinds[0]})
                            for s in dst.slots if s.key in q))

    def test_linux_keeps_native_engine(self):
        with mock.patch.object(plat, "IS_WIN", False):
            self.assertIsNone(models.convert("moss_native", native_params(), NATIVE_MODELS))
            self.assertEqual(models.hidden_aliases(), {})

    def test_aliases_on_windows(self):
        with mock.patch.object(plat, "IS_WIN", True):
            self.assertEqual(models.hidden_aliases(), {"moss_native": "moss"})
            self.assertTrue(models.runnable("moss"))
            self.assertFalse(models.runnable("moss_native"))


class Queue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patches = [mock.patch.object(tasks, "QUEUE_FILE", os.path.join(self.tmp, "queue.json")),
                        mock.patch.object(plat, "IS_WIN", True),
                        mock.patch.object(tasks, "_vram_gb", return_value=24),
                        mock.patch.object(settings, "gpu_list", return_value=["0"])]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        tasks._tasks = []
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _old_failed_task(self):
        return {"id": "1109180532_0fcf", "name": "mylora_moss_v1_huaf", "model_type": "moss_native",
                "models": dict(NATIVE_MODELS), "params": native_params(), "sample": {}, "output_dir": self.tmp,
                "status": "failed", "message": "训练器内部错误",
                "error": {"title": "训练器内部错误", "reason": "MOSS 专用引擎只支持 Linux"}}

    def test_load_migrates_failed_task(self):
        with open(tasks.QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump([self._old_failed_task()], f)
        tasks.load()
        t = tasks.get("1109180532_0fcf")
        self.assertEqual(t["model_type"], "moss")
        self.assertEqual(t["migrated_from"], "moss_native")
        self.assertEqual(t["status"], "stopped")   # 不再显示旧报错，点「从头重新训练」就能练
        self.assertIsNone(t["error"])
        self.assertIn("sd-scripts", t["message"])
        with open(tasks.QUEUE_FILE, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["model_type"], "moss")   # 写回了 queue.json

    def test_requeue_migrates(self):
        tasks._tasks = [self._old_failed_task()]
        t = tasks.requeue("1109180532_0fcf")
        self.assertEqual(t["model_type"], "moss")
        self.assertEqual(t["status"], "queued")
        self.assertFalse(t["params"]["convrot_int8"])   # 24 GB 档：DiT 用 bf16

    def test_add_with_native_draft(self):
        draft = {"model_type": "moss_native", "params": {"dataset_dir": r"X:\data\huaf", "output_name": "x_moss_v1"},
                 "models": dict(NATIVE_MODELS), "output_dir": self.tmp}
        t = tasks.add(draft)
        self.assertEqual(t["model_type"], "moss")
        self.assertEqual(t["params"]["output_name"], "x_moss_v1")
        self.assertIn("convrot_int8", t["params"])

    def test_runnable_task_untouched(self):
        t = {"id": "a", "model_type": "moss", "params": {"learning_rate": 3e-4}, "models": {}, "status": "failed"}
        tasks._tasks = [t]
        self.assertEqual(tasks._migrate(t), "")
        self.assertEqual(t["params"], {"learning_rate": 3e-4})
        self.assertEqual(t["status"], "failed")


if __name__ == "__main__":
    unittest.main()
