from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import build_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = build_manifest(cfg, max_records=args.max_records)
    out = Path(cfg["data"]["manifest_path"])
    if args.max_records is not None:
        out = out.with_name(out.stem + f"_first{args.max_records}" + out.suffix)
    save_manifest(manifest, out)

    print(f"manifest: {out}")
    print(manifest["summary"])
    print("discard reasons:", manifest["summary"]["discard_reasons"])
    kept = [r for r in manifest["records"] if r["status"] == "keep"]
    print(f"kept rows: {sum(r['n_rows'] for r in kept)}")
    print(f"kept raw windows: {sum(r['n_windows_raw'] for r in kept)}")


if __name__ == "__main__":
    main()
