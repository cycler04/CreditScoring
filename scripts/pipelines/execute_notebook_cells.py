#!/usr/bin/env python3
"""Execute notebook code cells sequentially for lightweight local smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import nbformat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    notebook = nbformat.read(args.notebook, as_version=4)
    namespace: dict[str, Any] = {"display": print}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        print(f"Executing cell {index}")
        compiled = compile(
            cell.source,
            f"{args.notebook}:cell-{index}",
            "exec",
        )
        exec(compiled, namespace)


if __name__ == "__main__":
    main()
