# -*- coding: utf-8 -*-
"""run_marimo.py — 按 marimo 运行时语义执行一个 notebook 的每个 cell。

marimo cell 形如 `def _(dep1, dep2): ...; return a, b`。本脚本：
  1) 原样 exec 每个 cell 的 `def`(顶层 return 合法)；
  2) 按依赖参数从共享命名空间 ns 传值调用；
  3) 把 return 的每个名字写回 ns 供后续 cell 使用。
从而在真实运行时下暴露每个 cell 的报错。用法:
    python run_marimo.py notebook.py
"""
import ast
import copy
import sys
import traceback
from pathlib import Path


def _deps(node):
    a = node.args
    pos = [x.arg for x in list(a.posonlyargs) + list(a.args)]
    return pos


def _return_names(node):
    for stmt in node.body:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            v = stmt.value
            items = v.elts if isinstance(v, (ast.Tuple, ast.List)) else [v]
            out = []
            for it in items:
                if isinstance(it, ast.Name):
                    out.append((it.id, None))
                elif isinstance(it, (ast.Tuple, ast.List)):
                    out.append((None, [e.id for e in it.elts if isinstance(e, ast.Name)]))
            return out
    return []


def run_cells(filepath):
    src = Path(filepath).read_text(encoding="utf-8")
    tree = ast.parse(src)
    cells = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_cell = any(
                isinstance(d, ast.Attribute) and d.attr == "cell"
                or isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                and d.func.attr == "cell"
                for d in n.decorator_list
            )
            if is_cell:
                cells.append(n)

    ns = {"__name__": "<marimo-run>"}
    ok = 0
    for i, n in enumerate(sorted(cells, key=lambda c: c.lineno), 1):
        fn_name = n.name
        label = f"cell#{i} (line {n.lineno})"
        try:
            node = ast.fix_missing_locations(copy.deepcopy(n))
            node.decorator_list = []  # 手动驱动调用，剥离 @app.cell 装饰器
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, "<cell>", "exec"), ns, ns)
            fn = ns[fn_name]
            kw = {d: ns[d] for d in _deps(n) if d in ns}
            missing = [d for d in _deps(n) if d not in ns]
            if missing:
                raise NameError(f"缺少依赖: {missing}")
            ret = fn(**kw)
            names = _return_names(n)
            if names:
                vals = ret if isinstance(ret, tuple) else (ret,)
                idx = 0
                for name, group in names:
                    if group is None:  # 单个名字
                        ns[name] = vals[idx]
                        idx += 1
                    else:  # (a, b) 简写
                        sub = vals[idx]
                        for g in group:
                            ns[g] = sub[group.index(g)]
                        idx += 1
            shown = ", ".join((nm for nm, g in names if g is None)) or "·"
            print(f"  [OK]   {label}  ->  {shown}")
            ok += 1
        except Exception:
            print(f"  [FAIL] {label}")
            print(traceback.format_exc())

    print(f"\n==> {ok}/{len(cells)} cells executed OK  ({filepath})")


if __name__ == "__main__":
    run_cells(sys.argv[1])