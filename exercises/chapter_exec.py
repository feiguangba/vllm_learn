# -*- coding: utf-8 -*-
"""chapter_exec.py — 一键「重新生成 + 执行 + 自检」一章所有 notebook。

用法:
    D:\\uv_envs\\uv_cuda\\Scripts\\python.exe chapter_exec.py ch01

对 chXN:
  1) 运行 chXN/_build/ 下所有 app/build_*.py(重建 ipynb 与 app);
  2) 对新生成的 *.ipynb 逐个 nbconvert --execute(用 uv_cuda kernel);
  3) 扫描每个 notebook 的 error cell,汇总报告。

这保证「补讲解 + 接真实 CUDA 数据」后的每一课都真正可执行、数据可复现。
"""
import os
import subprocess
import sys
from pathlib import Path

EXERCISES = r"D:\Project\21-Cpp_learn\explore\minivllm\exercises"
PY = r"D:\uv_envs\uv_cuda\Scripts\python.exe"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONPATH"] = EXERCISES


def find_build_scripts(ch_dir: Path):
    bd = ch_dir / "_build"
    if bd.exists():
        scripts = sorted(bd.glob("build_*.py"))
        if scripts:
            return scripts
        scripts = sorted(bd.glob("gen_*.py"))
        if scripts:
            return scripts
    # 兼容 gen_*.py 直接放在章节根目录(ch02)
    return sorted(ch_dir.glob("gen_*.py"))


def find_new_notebooks(ch_dir: Path):
    return sorted(ch_dir.glob("*.ipynb"))


def run(cmd, cwd, timeout=600):
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def execute_notebook(ipynb, timeout=600):
    return run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook",
                "--execute", "--inplace", str(ipynb),
                "--ExecutePreprocessor.kernel_name=uv_cuda",
                f"--ExecutePreprocessor.timeout={timeout}"], ipynb.parent)


def scan_errors(ipynb):
    import nbformat
    nb = nbformat.read(ipynb, as_version=4)
    errs = []
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and c.get("outputs"):
            for o in c["outputs"]:
                if o.get("output_type") == "error":
                    errs.append((i, o.get("ename", "?"), o.get("evalue", "")))
    return errs


def main(chapter: str):
    ch_dir = Path(EXERCISES) / chapter
    if not ch_dir.exists():
        print(f"!! 章节不存在: {chapter}")
        sys.exit(1)

    scripts = find_build_scripts(ch_dir)
    print(f"[{chapter}] 重建脚本 {len(scripts)} 个")
    for s in scripts:
        code, out = run([PY, "-s", "-X", "utf8", str(s)], ch_dir)
        tail = out.strip().splitlines()[-3:]
        tag = "OK" if code == 0 else "FAIL"
        print(f"  [{tag}] {s.name}")
        if code != 0:
            print(out[-1500:])

    nbs = find_new_notebooks(ch_dir)
    print(f"[{chapter}] 待执行 notebook {len(nbs)} 个")
    all_fail = []
    for ipynb in nbs:
        code, out = execute_notebook(ipynb)
        errs = scan_errors(ipynb) if code == 0 else [("kernel", "nbconvert", "execution crashed")]
        status = "OK" if (code == 0 and not errs) else "FAIL"
        print(f"  [{status}] {ipynb.name}  error_cells={len(errs)}")
        for (i, en, ev) in errs[:5]:
            print(f"      cell#{i} {en}: {ev[:120]}")
        if status == "FAIL":
            all_fail.append((ipynb.name, errs, out[-800:]))
    print("=" * 60)
    if all_fail:
        print(f"!! {len(all_fail)} 个 notebook 有错误")
        for n, errs, out in all_fail:
            print(f"  -- {n} --")
            print(out)
        sys.exit(2)
    print(f"[{chapter}] 全部通过 ✅")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])