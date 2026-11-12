# 给改这个仓库的 AI 看的说明

WWY LoRA 训练器（作者 WANG / wwy）。用户通过训练器设置里的「一键更新」从 GitHub 的 main 拿更新，所以推到 main 的东西会直接到所有用户手上。

## 版本号：每次更新都走一位（必做）

- **每次把改动推到 main，`version.json` 的 `version` 末位 +1**：0.1.2 → 0.1.3 → … → 0.1.9 → 0.1.10（末位一直往上加，不进位）。前两位只有 WANG 明确要求时才改。
- **一次更新只升一位**：同一次推送里有好几个提交，也只升一次。
- 用脚本改，不要手改：`python scripts/bump_version.py`。它会把末位 +1、把 `updated_at` 改成今天（日本时间）；如果这次更新已经升过（比 GitHub 上 main 的版本号大），它不会再升。
- 版本号改动放进这次的最后一个提交，或者单独提交一个「版本号 0.1.3」。
- 不要动 `version.json` 里的 `engines`（测试过的训练引擎版本，单独管理），也不要改 `commit.txt`（GitHub 打包时自动填提交号）。

## 推送前

- 跑测试：`python -m unittest discover -s tests`，全部通过再推。
- 「一键更新」的压缩包方式只复制 `app/`、`web/`、`scripts/` 和根目录下的几个固定文件（见 `app/updater.py` 的 `update_self` 里的列表）。根目录新加的文件到不了用户手上，要放进这几个文件夹，或者把文件名加进那个列表。
- 后端改了要重启训练器才生效（界面顶部会提示重启）；前端文件刷新页面就会重新加载。

## 约定

- 界面文字、提示、代码注释都用中文，写给不懂代码的用户看：说清楚发生了什么、该怎么办。
- Windows 整合包自带 Python 3.13；Linux 要求 3.10+，后端代码要兼容 3.10。
- `app/captionutil.py` 会被训练环境里的子进程（`app/workers/`）导入，只能用标准库。
- 标签编辑的拆分规则（`captionutil.parse_caption` / `norm_tag`）在 `web/js/editor.js` 里有一份一样的前端实现，改一边要同步改另一边。
