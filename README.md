# WWY LoRA 训练器

> **Linux / 云主机**：`./install.sh` 装环境，`./start.sh` 启动，详见 [README-Linux.md](README-Linux.md)。
> Windows 照旧双击 `启动LoRA训练器.bat`。

和 WWY 启动器同一套操作逻辑和界面风格的 LoRA 训练器。目前支持 **Anima**、**Krea 2**、**SDXL**（Illustrious / NoobAI / Pony 等）和 **MOSS-Image**，训练引擎是 kohya-ss 的官方实现（MOSS-Image 还可以切换到专用引擎）：

| 底模 | 引擎 | 训练脚本 |
|---|---|---|
| Anima（Cosmos-Predict2 2B + Qwen3-0.6B） | [sd-scripts](https://github.com/kohya-ss/sd-scripts) | `anima_train_network.py` + `networks.lora_anima` |
| MOSS-Image（5B 单流 DiT + Qwen3.5-9B + FLUX.2 VAE；Base / SFT，官方分片或 ComfyUI 单文件，bf16 / int8 ConvRot） | [sd-scripts](https://github.com/kohya-ss/sd-scripts) | `moss_image_train_network.py` + `networks.lora_generic`（官方预训练时间步分布、ConvRot INT8、8 GB 显卡用文本编码器分层卸载） |
| MOSS-Image（同上，模型设置里切换引擎；bf16 单文件 + 官方 VAE；**仅 Linux**，Windows 下不显示） | MOSS 专用引擎（[waw1w1/moss_image_trainer](https://github.com/waw1w1/moss_image_trainer)，已获作者同意整合在 `moss_trainer/`） | `moss_trainer.cli cache` → `train`（LyCORIS：LoRA / LoCon / LoHa / LoKr） |
| SDXL（单文件大模型，含 Illustrious / NoobAI / Pony / Animagine 等微调） | [sd-scripts](https://github.com/kohya-ss/sd-scripts) | `sdxl_train_network.py` + `networks.lora`（v-pred 模型自动开 v-parameterization + zero terminal SNR） |
| Krea 2（12B MMDiT + Qwen3-VL-4B） | [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) | `krea2_cache_latents.py` → `krea2_cache_text_encoder_outputs.py` → `krea2_train_network.py` |

## 使用

双击 `启动LoRA训练器.bat`。第一次会自动下载便携版 Python（约 30 MB，不改系统），然后在浏览器打开界面（默认 http://127.0.0.1:28100）。
第一次训练前到「训练器设置 → 安装训练环境」装 PyTorch 和两个引擎（下载约 4 GB）。

侧边栏八个页面：

1. **启动训练**：当前配置摘要、保存位置、开始 / 加入队列 / 停止；多个任务排队依次训练；运行日志、loss 曲线；配置存档。**冻结**：保存完整训练状态后停下、释放显存（比如要玩游戏），点「继续」从那一步接着练；停止 / 失败的任务可以**一键续训**（有训练状态用状态，没有就从最后一轮的 LoRA 接着练）。目前 Anima、SDXL 支持。练好的 LoRA 一键**送到 WWY 启动器**（共享模型库按「底模/类型」放，或某个实例），示例图当预览图，触发词写进模型信息。
2. **模型设置**：选 Anima、Krea 2、SDXL 或 MOSS-Image，自动在电脑里找 WebUI / ComfyUI / WWY 启动器共享模型库，按文件内容（不是文件名）识别主模型、文本编码器、VAE，优先选 base 模型；SDXL 大模型很多，主模型用可搜索的选择器，按「官方底模 / Illustrious 系 / NoobAI 系 / v-pred / Pony 系 / 其他」分组（参考文件名、Civitai 信息和文件头里的 v-pred 标记）；下拉框只列出这个底模能用的文件，也可以自选；找不到时一键下载官方文件。
3. **高级参数**：按显存一键套用；梯度检查点在最上面；保存名称、训练集、每张图重复次数（不用改文件夹名）、轮数 / 总步数、保存间隔、批量、dim、alpha；实时计算总步数；学习率；其它高级设置。
4. **图片爬取**：从 Danbooru 按标签爬训练图并生成标签。标签框接入 [Danbooru 中英字典](https://github.com/wangjue520/danbooru-zh-dict)：中文 / 英文 / 别名都能搜，只能添加 D 站正式标签（别名自动换成正式名，拼错会提示正确写法）；过滤预设「人物 LoRA」（只要单人图）、「画风 LoRA」（去掉草图、线稿、截图等）、「通用去噪」。名额不够时自动把一部分标签、排序、黑名单改到本地处理，不会因为「标签太多」失败；开始前检查标签是否存在并给出纠正建议；可按分数 / 收藏全局精确排序、限制最小分辨率和分级；爬完一键设为训练集。
5. **标签反推**：WD14 / 本地自然语言（ToriiGate 0.5、JoyCaption Beta One、Qwen3-VL abliterated）/ WD14+自然语言混合 / API（OpenAI 兼容）；触发词。
6. **标签编辑**：反推完自动打开那组训练集；**触发词**：自动判断主触发词（出现最多的角色）、角色特征、服装（按出现比例或一张典型图片），主触发词放在每个标签文件最前面、其余标签顺序不变，可选精简掉特征标签；标签统计、筛选、批量增删改、拖动排序、自动补全，批量操作可撤销。
7. **训练中测试**：每 N 步用固定提示词和种子出一组图，表格对比各步数效果（Krea 2 可用 Turbo 8 步出图）。
8. **训练器设置**：一键更新、训练环境安装 / 修复、国内 / 国外网络、端口、界面风格、显卡选择、训练集文件夹同步（爬图保存位置 / 标签反推 / 标签编辑 / 高级参数的训练集改一处全跟着变）等。

### 自动处理（不用管）

- 下载线路自动选择：每次下载模型 / 反推模型 / 字典 / pip 组件 / 引擎源码前，并行测速所有线路——下载源（HF 镜像 hf-mirror、HF 官方；GitHub 直连和几个加速代理；清华 / 阿里云 / 中科大 / 官方 pip 源）× 连接方式（直连、VPN 的系统代理、「训练器设置 → 网络」里填的代理）——挑最快的；连不上的线路（比如 VPN 关了但系统代理还开着）自动跳过。下载中出错会自动换下一条线路，已下好的部分不用重下。设置页可以一键测速看各线路速度。
- 大文件多连接下载：8 条连接分块下载，国内单连接限速时快好几倍，支持断点续传。

- 更新同步：训练器、训练引擎（sd-scripts / musubi-tuner）、训练环境的依赖三样版本绑定。`version.json` 的 `engines` 记着这个版本的训练器测试过的引擎提交号；「一键更新」先更新训练器本体，再用新代码把引擎切到对应版本、装好新的依赖、做环境检查。训练器不管怎么更新的（一键更新 / git pull / 手动覆盖），启动时发现引擎版本或依赖清单对不上就自动在后台同步，同步期间队列里的训练会等着。发布新版本前在训练环境里测好引擎后，用 `python -c "import envsetup; envsetup.pin_current()"`（在 app 目录下）把当前引擎版本写进 `version.json`。
- 透明底图片：爬图完成后和每次训练前自动改成白底（训练引擎直接丢掉透明通道会变成黑底），原图备份在 `data/透明底原图备份/`。开了「透明通道当 loss 遮罩」时不改。

### 报错处理

开训前自动体检（训练集空文件 / 损坏图片 / 缺标签、模型文件是否完整且选对、显存是否被占、硬盘空间）。只有练不起来的致命问题才会拦下等你处理；显存被占、测试提示词和标法对不上这类只写进日志提醒，照常开训。
训练失败时按日志识别报错，给出中文原因和可选的修复动作（结束占显存的程序、调低显存占用、重新下载损坏的模型、清理训练集、安装缺失组件、学习率减半……）。
**所有修复都会先列出要做的事（要删的文件、要结束的进程）并等你确认后才执行**；没法自动修的会说明原因和手动处理方法。

## 目录

```
启动LoRA训练器.bat   唯一入口
app/                 后端（只用 Python 标准库）
  models/            底模注册表：每个底模一个文件
  workers/           在训练环境里运行的子进程（反推）
web/                 界面
scripts/             便携 Python 下载脚本
运行时生成：python/ 便携 Python · env/ 训练环境 · engines/ 训练引擎 · models/ 下载的模型 · data/ 设置和队列 · output/ 默认保存位置
```

## 接入新的底模

在 `app/models/` 新建一个文件，继承 `ModelSpec`：

- `slots`：需要哪些模型文件（主模型 / 文本编码器 / VAE …），每个槽位接受哪些 `kind`（`app/safetensors_meta.py` 按文件内容识别），找不到时的官方下载地址
- `score()` / `accept()`：自动选择的优先级、哪些文件能用
- `params()`：参数表单（`common.py` 里有通用字段，按需覆盖默认值、追加专属字段）
- `vram_presets()`：显存预设
- `build_steps()`：生成训练命令（可以多步，例如先缓存再训练）

然后在 `app/models/__init__.py` 里注册。界面会自动出现新底模的卡片、模型槽位、参数表单和显存预设。
如果新底模需要新的模型结构识别，在 `safetensors_meta.classify()` 里加一个分支。
