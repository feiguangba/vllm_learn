"""
nb_builder.py — VLLM_learn 笔记本生成工具
用法:  import sys; sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises")
       from nb_builder import Notebook
       nb = Notebook("标题", emoji="🚀")
       nb.md("# 第 1 课 ...", "段落...")
       nb.code("import torch ...", "注释说明")
       nb.save("01_chapter/file.ipynb")
"""
from pathlib import Path
import nbformat as nbf

class Notebook:
    def __init__(self, title: str, subtitle: str = "", emoji: str = "📘", chapter: str = ""):
        self.title = title
        self.chapter = chapter
        self.nb = nbf.v4.new_notebook()
        self.nb.metadata = {
            "kernelspec": {
                "display_name": "Python 3 (uv_cuda)",
                "language": "python",
                "name": "uv_cuda",
            },
            "language_info": {"name": "python", "version": "3.12"},
        }
        self.cells = []
        cover = [f"# {emoji} {title}"]
        if subtitle:
            cover.append(f"\n> {subtitle}")
        if chapter:
            cover.append(f"\n**章节**: {chapter}")
        cover.append(
            "\n---\n"
            "\n> **📚 本笔记本属于《VLLM_learn: 图解 vLLM 推理引擎》系列**"
            "\n> 风格参照《鸢尾花书》: 重图解、重直觉、循序渐进、动手实操"
        )
        self.md("\n".join(cover))

    def md(self, *blocks):
        """加一个或多个 markdown 块"""
        for b in blocks:
            self.cells.append(nbf.v4.new_markdown_cell(b))
        return self

    def code(self, src: str, note: str = ""):
        """加一个代码块,note 为代码上方的说明(markdown)"""
        if note:
            self.cells.append(nbf.v4.new_markdown_cell(note))
        self.cells.append(nbf.v4.new_code_cell(src))
        return self

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.nb.cells = self.cells
        with open(p, "w", encoding="utf-8") as f:
            nbf.write(self.nb, f)
        print(f"[ok] {p}")


def chapter_cover(nb: Notebook, objectives: list, toc: list, links: list = None):
    """章节封面: 学习目标 + 目录 + 参考链接"""
    lines = ["## 🎯 学习目标", ""]
    lines += [f"- {o}" for o in objectives]
    lines += ["", "## 📑 本课目录", ""]
    for i, (label, desc) in enumerate(toc, 1):
        lines.append(f"{i}. **{label}** — {desc}")
    if links:
        lines += ["", "## 🔗 参考资料", ""]
        lines += [f"- [{t}]({u})" for t, u in links]
    nb.md("\n".join(lines))


def wrapup(nb: Notebook, summary: list, practice: list, links: list = None):
    """结尾: 小结 + 练习 + 参考"""
    lines = ["## ✅ 本课小结", ""]
    lines += [f"- {s}" for s in summary]
    lines += ["", "## ✍️ 动手练习", ""]
    for i, p in enumerate(practice, 1):
        lines.append(f"{i}. {p}")
    if links:
        lines += ["", "## 🔗 延伸阅读", ""]
        lines += [f"- [{t}]({u})" for t, u in links]
    lines += ["", "---", "> 🎉 恭喜完成本课!下一课见!"]
    nb.md("\n".join(lines))