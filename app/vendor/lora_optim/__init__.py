"""训练器自带的额外优化器。训练子进程的 PYTHONPATH 包含 app/vendor，所以优化器类型填「lora_optim.Automagic」即可。

Automagic / EmoSens 都会自己调整学习率，训练引擎记到日志里的「学习率」只是调度器的起始值，看不出真实变化。
这里在它们每一步之后把实际的平均学习率追加到训练器指定的文件（环境变量 LORA_ACTUAL_LR_FILE），
训练页据此画「实际学习率」和「每轮学习率」。多卡时只有主进程写。"""
import os

from .automagic import Automagic as _Automagic


class _ActualLrLog:
    """混入类：step() 之后记录实际学习率（每 N 步一行：步数\\t学习率）。"""

    _lr_every = 1

    def _lr_log_init(self):
        self._lr_file = os.environ.get("LORA_ACTUAL_LR_FILE")
        if os.environ.get("RANK", "0") not in ("0", ""):
            self._lr_file = None   # 多卡时只让主进程写
        self._lr_steps = int(os.environ.get("LORA_ACTUAL_LR_BASE") or 0)   # 续训时从已经练过的步数接着记

    def _actual_lr(self):
        raise NotImplementedError

    def _lr_log_step(self):
        if not getattr(self, "_lr_file", None):
            return
        self._lr_steps += 1
        if self._lr_steps % self._lr_every:
            return
        try:
            v = float(self._actual_lr())
            with open(self._lr_file, "a", encoding="utf-8") as f:
                f.write(f"{self._lr_steps}\t{v:.6e}\n")
        except Exception:
            pass   # 记录失败不能影响训练


class Automagic(_ActualLrLog, _Automagic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lr_log_init()

    def _actual_lr(self):
        return self.get_avg_learning_rate()

    def step(self, closure=None):
        loss = super().step(closure)
        self._lr_log_step()
        return loss

    def load_state_dict(self, state_dict, strict=True):
        """续训：原版把恢复的 lr_mask 留在 CPU（accelerate 读状态时 map_location=cpu），第一步就报设备不一致。
        这里把每个参数的 lr_mask 等状态挪到参数所在的设备上。"""
        super().load_state_dict(state_dict, strict)
        import torch
        from .optimizer_utils import Auto8bitTensor
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                m = st.get("lr_mask")
                if isinstance(m, Auto8bitTensor) and m.quantized.device != p.device:
                    m.quantized = m.quantized.to(p.device)
                    if torch.is_tensor(m.scale):
                        m.scale = m.scale.to(p.device)
                for k in ("avg_lr", "last_polarity", "exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq"):
                    if torch.is_tensor(st.get(k)) and st[k].device != p.device:
                        st[k] = st[k].to(p.device)


def __getattr__(name):
    # EmoSens 导入时会改 Tensor.backward，按需加载：选了别的优化器时完全不碰它
    if name == "EmoSens":
        from .emosens import EmoSens as _EmoSens

        class EmoSens(_ActualLrLog, _EmoSens):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._lr_log_init()

            def _actual_lr(self):   # EmoSens 每步把算出的 emoPulse 写回 param_groups 的 lr
                lrs = [g["lr"] for g in self.param_groups]
                return sum(lrs) / len(lrs)

            def step(self, closure=None):
                loss = super().step(closure)
                self._lr_log_step()
                return loss

        globals()["EmoSens"] = EmoSens
        return EmoSens
    raise AttributeError(name)
