# -*- coding: utf-8 -*-
"""双卡时选了 A 卡却跑在 B 卡上（群友报：笔记本 + 外接显卡）。

原因：设置页的显卡编号来自 nvidia-smi（按 PCI 总线排），CUDA 默认按 FASTEST_FIRST（最快的排第一）编号，
CUDA_VISIBLE_DEVICES=0 指的是 CUDA 眼里的 0 号。两张卡里 PCI 总线靠后的那张更快时（外接的新卡、独显 + 显卡坞），
两套编号正好相反。修复：所有子进程统一 CUDA_DEVICE_ORDER=PCI_BUS_ID。

运行：python -m unittest discover -s tests"""
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import gpu  # noqa: E402
import paths  # noqa: E402
import settings  # noqa: E402


class ChildEnvOrder(unittest.TestCase):
    def test_pci_bus_order_even_if_user_set_fastest_first(self):
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}), \
                mock.patch.object(settings, "load", return_value=dict(settings.DEFAULTS, gpu_id="1", gpu_ids=[])):
            env = settings.child_env()
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_training_step_override_keeps_order(self):
        # tasks._run_step 用 child_env({"CUDA_VISIBLE_DEVICES": ...}) 覆盖显卡，排序方式不能丢
        with mock.patch.object(settings, "load", return_value=dict(settings.DEFAULTS)):
            env = settings.child_env({"CUDA_VISIBLE_DEVICES": "1,0"})
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,0")

    def test_no_subprocess_sets_visible_devices_without_child_env(self):
        # 所有设置 CUDA_VISIBLE_DEVICES 的地方都必须经过 settings.child_env（它负责 CUDA_DEVICE_ORDER）
        app = paths.APP
        for root, _, files in os.walk(app):
            if "vendor" in root:
                continue
            for fn in files:
                if not fn.endswith(".py") or fn == "settings.py":
                    continue
                with open(os.path.join(root, fn), encoding="utf-8") as f:
                    src = f.read()
                for line in src.splitlines():
                    if "CUDA_VISIBLE_DEVICES" in line and "=" in line and "child_env" not in line \
                            and not line.strip().startswith("#") and "split" not in line:
                        self.fail(f"{fn} 里绕过 child_env 设置了 CUDA_VISIBLE_DEVICES：{line.strip()}")

    def test_describe(self):
        fake = {"all": [{"index": "0", "name": "NVIDIA GeForce RTX 4060 Laptop GPU", "total_gb": 8, "bus": "00000000:01:00.0"},
                        {"index": "1", "name": "NVIDIA GeForce RTX 4090", "total_gb": 24, "bus": "00000000:3C:00.0"}]}
        with mock.patch.object(gpu, "info", return_value=fake):
            self.assertEqual(gpu.describe(["1"]), "使用显卡：1 号 NVIDIA GeForce RTX 4090（PCI 00000000:3C:00.0）")


_PROBE = ("import json, torch; p = torch.cuda.get_device_properties(0); "
          "print(json.dumps({'name': p.name, 'bus': '%04x:%02x:%02x' % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id)}))")


def _norm_bus(b):
    # nvidia-smi：00000000:01:00.0；torch：0000:01:00
    dom, bus, dev = b.split(":")[0][-4:], b.split(":")[1], b.split(":")[2].split(".")[0]
    return f"{dom}:{bus}:{dev}".lower()


@unittest.skipUnless(paths.env_python() and gpu.info(), "没有训练环境或 N 卡")
class RealHardware(unittest.TestCase):
    """真机：按设置页的编号启动训练环境的 python，CUDA 里的 0 号必须是 nvidia-smi 里同一张卡（看 PCI 总线号）。"""

    def test_each_index_maps_to_same_card(self):
        for card in gpu.info(max_age=0)["all"]:
            with mock.patch.object(settings, "load", return_value=dict(settings.DEFAULTS, gpu_id=card["index"], gpu_ids=[])):
                env = settings.child_env()
            out = subprocess.run([paths.env_python(), "-c", _PROBE], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            got = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertEqual(got["bus"], _norm_bus(card["bus"]), f"{card['index']} 号：nvidia-smi 是 {card['name']}，CUDA 用的是 {got['name']}")


if __name__ == "__main__":
    unittest.main()
