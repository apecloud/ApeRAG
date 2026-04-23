"""Export ApeRAG OpenAPI specs from the FastAPI app without starting a server.

The exported specs are build artifacts:
- full spec: internal web/admin typing and generated clients
- public spec: future public SDK/API docs with internal runtime paths removed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aperag.app import app  # noqa: E402
from aperag.openapi_spec import build_full_openapi_spec, filter_public_openapi  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def export_specs(*, full_output: Path, public_output: Path) -> None:
    app.openapi_schema = None
    full_spec = build_full_openapi_spec(app)
    public_spec = filter_public_openapi(full_spec)
    _write_json(full_output, full_spec)
    _write_json(public_output, public_spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ApeRAG OpenAPI specs from the FastAPI app.")
    parser.add_argument("--full-output", default="openapi.full.json", help="Full spec output path.")
    parser.add_argument("--public-output", default="openapi.public.json", help="Public spec output path.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that OpenAPI export succeeds without writing to the requested paths.",
    )
    args = parser.parse_args()

    if args.check:
        with TemporaryDirectory() as tmp:
            export_specs(
                full_output=Path(tmp) / "openapi.full.json",
                public_output=Path(tmp) / "openapi.public.json",
            )
        return

    export_specs(full_output=Path(args.full_output), public_output=Path(args.public_output))


if __name__ == "__main__":
    main()
