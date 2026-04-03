---
name: 从周刊草稿更新周刊
description: 在用户指定用新一期的周刊草稿来创作本周的周刊报告时生效
---

本技能将草稿转为当周正式周刊（中英双语），并统一下载论文 PDF、提取关键图、内联插图、更新首页目录。

前置约定
- 操作系统：Windows
- Python 虚拟环境：.venv（路径：.venv\Scripts\python.exe）
- 目录规范：
  - 中文版：2026\第{n}期.md
  - 英文版：2026\第{n}期_en.md
  - 本期资源：Resource\第{n}期\（PDF、图片、figures_metadata.json）
- 徽章与链接：统一使用 shields.io arXiv 徽章 + 本地 PDF 下载链接

步骤
1) 预览草稿
- 打开根目录下的“第{n}期草稿.md”，通读确定主题结构与选题清单。

2) 参考上一期版式
- 打开上一期中文正式稿（例如第十期：2026\第十期.md），复用：
  - 顶部中英切换、期标题、关键词行
  - 目录与分节结构（连续编号）
  - 每篇论文的标准化区块（标题、徽章、PDF、本期说明要点）

3) 生成本期中文正式稿
- 按上一期版式新建：2026\第{n}期.md（避免 AI 腔，一律客观凝练）。
- 每篇包含：
  - 论文标题与说明
  - arXiv 徽章与本地 PDF 下载入口（../Resource/第{n}期/xxx.pdf）
  - **丰富多维度的内容扩写（不可干瘪）：** 每篇论文的解读必须包含以下子模块，并使用加粗标题：
    - **研究背景 (Motivation)**：解决了什么行业痛点或理论空缺。
    - **核心机制/方法 (Methodology)**：具体的架构、算法或系统设计细节。
    - **实验与结果 (Results)**：具体的数据集、提升指标或工程效能。
    - **核心启示 (Insights)**：该工作对未来方向、底层理论或工程落地的深层影响。

4) 优先使用本地 PDF，必要时补充下载
- 创建目录：Resource\第{n}期\
- 如 PDF 已存在，跳过下载；若缺失，使用 PowerShell 下载：

```powershell
New-Item -ItemType Directory -Path 'Resource\第{n}期' -Force | Out-Null
Invoke-WebRequest -Uri 'https://arxiv.org/pdf/{arxiv_id}.pdf' -OutFile 'Resource\第{n}期\{Name}.pdf'
```

5) 提取插图与生成元数据
- 本仓库自带脚本 utils\arxiv_pdf_download.py 已支持：
  - 传入 --pdf 且不带 --term 时，提取所有标注为 “Figure/Fig.” 的图片
  - 输出图片与 figures_metadata.json 至 --out 指定目录
- 在本期目录运行（优先使用本地 PDF）：

```powershell
.venv\Scripts\python.exe utils\arxiv_pdf_download.py ^
  --pdf Resource\第{n}期\A.pdf Resource\第{n}期\B.pdf ... ^
  --out Resource\第{n}期
```

6) 在中文稿内联插图（每篇 1–2 张）
- 在对应小节加入：

```html
<div align="center">
  <img src="../Resource/第{n}期/{paper}_Figure1.png" width="80%">
  <br>
  <em>图：简短说明。</em>
</div>
```
- 选择标准：优先“总体框架/流程图/关键对比图”。若无可用图，保持文本版式；如该论文有官方仓库或项目页且许可明确，可补充示意图。

7) 生成英文版
- 复制中文结构至 2026\第{n}期_en.md，保持相同的论文顺序与插图位置；标题/说明用英文精简表述。

8) 更新首页目录
- readme.md 与 readme_en.md 的 2026 存档表格首行加入本期：
  - 中文：No.{n} → 链接至 ./2026/第{n}期.md
  - English：No.{n} → 链接至 ./2026/第{n}期_en.md

9) 交叉检查
- 校验：所有图片、PDF、本地相对路径是否可点击；中英文顶部互链是否工作；目录锚点跳转是否正确。

附：常用命令速查
- 以本地 PDF 批量提取全部图：
```powershell
.venv\Scripts\python.exe utils\arxiv_pdf_download.py --pdf Resource\第{n}期\*.pdf --out Resource\第{n}期
```
- 提取指定图（例如 Figure 2）：
```powershell
.venv\Scripts\python.exe utils\arxiv_pdf_download.py --pdf Resource\第{n}期\A.pdf --term \"Figure 2\" --out Resource\第{n}期
```

说明
- 若个别选题无 arXiv 条目（或无图），可保留文本条目并注明来源/状态。
- 保持文件与路径命名一致性；避免泄露任何凭证；所有外链使用 https。***
