import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import math
    arr_a=np.array([1,2,3,4,5,6,])
    return arr_a, math


@app.cell
def _(arr_a):
    arr_a
    return


@app.cell
def _(mo):
    x=mo.ui.slider(1,15)
    return (x,)


@app.cell
def _(math, mo, x):
    mo.md(f"""
    $e^{x.value}={math.exp(x.value):0.3f}$
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
