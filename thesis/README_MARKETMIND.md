# MarketMind 论文目录说明

本目录接入自 `ustctug/ustcthesis`，用于在当前 MarketMind 仓库中管理毕业论文 LaTeX 源码。

## 模板来源

- GitHub: <https://github.com/ustctug/ustcthesis>
- 当前接入方式：复制模板内容到 `thesis/`，并移除了模板自身的 `.git` 元数据，使其作为 MarketMind 仓库的一部分管理。

## 本地编译

进入本目录后编译：

```bash
cd thesis
latexmk -xelatex main.tex
```

清理临时文件：

```bash
cd thesis
latexmk -c
```

## LaTeX 环境

模板需要 TeX Live / MacTeX / MiKTeX。macOS 推荐安装 MacTeX 或 BasicTeX。

当前自动安装情况：

- `mactex-no-gui`：下载时 CTAN 镜像多次中断，未完成。
- `basictex`：安装包已下载，但 macOS `.pkg` 安装需要管理员密码；当前非交互终端无法输入 sudo 密码，因此未完成安装。

可在本机终端手动执行：

```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
which xelatex
which latexmk
```

如果选择完整 MacTeX：

```bash
brew install --cask mactex-no-gui
eval "$(/usr/libexec/path_helper)"
```

## 写作入口

- 主文件：`main.tex`
- 论文信息配置：`ustcsetup.tex`
- 正文章节：`chapters/`
- 图片目录：`figures/`
- 参考文献：`bib/`
