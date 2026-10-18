"""Static checks: compile every module and list unused imports (AST-based)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "src" / "ccm", ROOT / "app", ROOT / "tests", ROOT / "scripts"]
FILES = [ROOT / "run_controls.py"] + [p for t in TARGETS for p in sorted(t.rglob("*.py"))]

BUILTIN_NAMES = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))

failures: list[str] = []
for path in FILES:
    if path.name.endswith("_version.py") or not path.exists():
        continue
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - build gate
        failures.append(f"{path}: SYNTAX ERROR {exc}")
        continue

    imported: dict[str, ast.AST] = {}
    lazy_future = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = (alias.asname or alias.name.split(".")[0])
                imported[name] = node
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                lazy_future = True
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                imported[name] = node

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    # attribute chains: also capture dotted roots
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            base = n
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                used.add(base.id)
    # strings used in annotations / docstrings are out of scope for this quick scan

    # Files that exist to re-export a public API re-import names on purpose
    # (marked `# noqa: F401`); they are skipped here.
    if path.name != "__init__.py":
        unused = sorted(set(imported) - used - BUILTIN_NAMES, key=str.casefold)
        if unused:
            failures.append(f"{path.relative_to(ROOT)}: unused import(s) -> {', '.join(unused)}")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print(f"STATIC CHECK OK - {len(FILES)} files, no syntax errors, no unused imports")
