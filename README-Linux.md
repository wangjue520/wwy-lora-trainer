# Linux 版说明（云主机 / 容器）

界面、训练流程、`app/models` 里的底模定义和 Windows 版完全一样，Windows 专有的部分换成了 Linux 的等价做法（都在 `app/plat.py`）。
`data/`、`output/`、`models/`、`engines/` 可以和 Windows 版共用；`env/` 不行，要在 Linux 上重装。

## 环境要求

- Linux x86_64（Ubuntu 20.04+ / Debian 11+；仙宫云、AutoDL 等镜像都可以）
- NVIDIA 驱动 ≥ 525（`nvidia-smi` 能正常输出）
- Python 3.10+（系统自带或 conda 都行）
- 磁盘：训练环境约 10 GB，模型另算；内存建议 ≥ 16 GB

## 快速开始

```bash
./install.sh                     # 装训练环境（venv + PyTorch + 两个引擎 + 反推依赖），可重复执行
./install.sh --torch cu128       # 指定 PyTorch 的 CUDA 版本
./start.sh                       # 后台启动，日志 trainer.log
./stop.sh                        # 停止
./run.sh                         # 前台运行（Ctrl+C 退出）
```

`install.sh` / `start.sh` 优先用 `python/bin/python`（项目自带的 Python），否则用 `python3`；
系统 Python 缺 `venv / ensurepip` 时会自动换 miniconda 等带 venv 的解释器。

训练器默认监听 `0.0.0.0:28100`（`PORT=28200 ./start.sh` 换端口，`LORA_TRAINER_EXTRA_PORTS=80` 再多监听一个端口）。
从自己电脑访问可以用 SSH 隧道：

```bash
ssh -N -L 28100:127.0.0.1:28100 -p <SSH端口> root@<SSH地址>
```

然后浏览器打开 `http://127.0.0.1:28100/`。

## 仙宫云

```bash
bash scripts/xgc/setup_xgc.sh
```

- 开机自动启动训练器（`/scripts/start.d/wwy_lora_trainer.sh`），同时监听 28100 和 80；
  控制台的「打开webui」按钮开的就是 80 端口。
- 仙宫云 OS 桌面：LoRA训练器、启动训练器、停止训练器、训练器日志、训练集、训练输出、模型文件。
  基础镜像自带的别的应用图标 / 启动项挪到 `/root/.xgc_old/`。
- 外网地址：`https://<实例ID>-28100.container.x-gpu.com/`（或 `-80`）。

## MOSS-Image 的两个训练引擎

模型设置里 MOSS-Image 卡片下面切换：

| | sd-scripts | MOSS 专用引擎（[waw1w1/moss_image_trainer](https://github.com/waw1w1/moss_image_trainer)，已获作者同意整合） |
|---|---|---|
| 主模型 / 文本编码器 | bf16、int8 ConvRot、官方分片都行 | 只认未量化 bf16 / fp16 单文件 |
| VAE | ComfyUI flux2-vae 或官方都行 | 只认官方 `vae/diffusion_pytorch_model.safetensors` |
| 网络类型 | LoRA / LoKr / LoHa | LoRA / LoCon / LoHa / LoKr |
| 冻结 / 续训、训练中换主模型和加速出图 | 支持 | 不支持 |

两个引擎共用训练环境 `env/`（专用引擎额外只需要 `lycoris-lora`，装环境时一起装）。
专用引擎第一次用会下载官方配置和分词器到 `models/moss_official`（几十 MB）。

## 和 Windows 版的区别

| 地方 | Windows | Linux |
|---|---|---|
| 入口 | `启动LoRA训练器.bat` | `run.sh` / `start.sh` / `stop.sh` / `install.sh` |
| 训练环境 | 整合包自带便携 Python | `python3` 建的 venv（`env/`） |
| 结束进程 | `taskkill /T /F` | 子进程在独立进程组，`killpg` 整组结束 |
| 取进程命令行 | PowerShell | `/proc/<pid>/cmdline` |
| 文件选择框 / 找 WebUI | 从盘符开始 | 从 `~`、`/root`、`/workspace`、`/data`、`/mnt` 开始 |
| 自动开浏览器 | 是 | 否 |

## 常见问题

- **`python3 -m venv` 失败**：`apt install -y python3-venv`，或 `PYTHON=/root/miniconda3/bin/python ./install.sh`。
- **搬了目录**：`env/` 里是绝对路径，删掉 `env/` 重新 `./install.sh`。
- **被墙的机器爬 Danbooru**：`LORA_DANBOORU_BASE` 指向镜像 / 反代（例如 `LORA_DANBOORU_BASE=http://127.0.0.1:18888 ./start.sh`）。
