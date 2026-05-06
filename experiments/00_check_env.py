# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from typing import List, Tuple

import requests


MODULES: List[Tuple[str, str]] = [
    ("numpy", "numpy"),
    ("networkx", "networkx"),
    ("requests", "requests"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("sentence_transformers", "sentence-transformers"),
    ("FlagEmbedding", "FlagEmbedding"),
]


def check_module(import_name: str, package_name: str) -> bool:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "未知")
        print(f"[OK] {package_name}: {version}")
        return True
    except Exception as exc:
        print(f"[MISS] {package_name}: {exc!r}")
        return False


def check_pip() -> None:
    print("\n# Python 环境")
    print(sys.executable)
    print(sys.version)
    print("\n# Pip 环境")
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "--version"], text=True)
        print(out.strip())
    except Exception as exc:
        print(f"pip 检查失败: {exc!r}")


def check_ollama() -> None:
    print("\n# Ollama")
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name") for m in data.get("models", [])]
        print("[OK] Ollama 可连接")
        print(json.dumps(models, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[MISS] Ollama 无法连接或尚未就绪: {exc!r}")


def main() -> None:
    check_pip()
    print("\n# Python 包")
    ok = True
    for import_name, package_name in MODULES:
        ok = check_module(import_name, package_name) and ok
    check_ollama()
    if not ok:
        print("\n缺少部分 Python 包。请运行：")
        print("python -m pip install -r requirements.txt")
        print("或者单独安装：")
        print("python -m pip install -U FlagEmbedding")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
