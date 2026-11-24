from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write LIBERO's plain-YAML runtime paths for this machine."
    )
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, default=Path("datasets/libero"))
    args = parser.parse_args()

    package_root = args.libero_root.expanduser().resolve() / "libero" / "libero"
    required = {
        "benchmark_root": package_root,
        "bddl_files": package_root / "bddl_files",
        "init_states": package_root / "init_files",
        "assets": package_root / "assets",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing LIBERO paths: " + ", ".join(missing))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: str(path) for key, path in required.items()}
    payload["datasets"] = str(args.datasets.expanduser().resolve())
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
