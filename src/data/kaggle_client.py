"""Kaggle dataset search and download, using the official ``kaggle`` package.

Authentication uses Kaggle's own mechanisms only (checked in this order by the Kaggle client):
  * ``KAGGLE_API_TOKEN`` environment variable, or ``~/.kaggle/access_token``  (new-style API token)
  * ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` environment variables                (legacy key)
  * ``~/.kaggle/kaggle.json`` (or ``$KAGGLE_CONFIG_DIR/kaggle.json``)         (legacy key file, chmod 600)

Credentials are never read into this project's config, logged or written anywhere by this module.
Downloads happen only for a slug the user names explicitly, and require ``confirm=True`` / ``--yes``.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import PROJECT_ROOT

log = logging.getLogger(__name__)


class KaggleAuthError(RuntimeError):
    pass


class KaggleClientError(RuntimeError):
    pass


SETUP_HELP = (
    "Kaggle credentials not found or rejected. Create an API token at https://www.kaggle.com/settings "
    "(API section), then either export KAGGLE_API_TOKEN=<token>, or export KAGGLE_USERNAME and KAGGLE_KEY, "
    "or save kaggle.json to ~/.kaggle/kaggle.json and run: chmod 600 ~/.kaggle/kaggle.json. "
    "If Kaggle is unreachable, use the manual import workflow in README.md (Datasets > Manual import)."
)


def credential_source() -> str | None:
    """Which credential mechanism is present (never returns the secret itself)."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "env:KAGGLE_API_TOKEN"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "env:KAGGLE_USERNAME/KAGGLE_KEY"
    cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    if (cfg_dir / "access_token").is_file():
        return f"file:{cfg_dir / 'access_token'}"
    if (cfg_dir / "kaggle.json").is_file():
        return f"file:{cfg_dir / 'kaggle.json'}"
    return None


def _safe_slug(slug: str) -> str:
    parts = slug.strip().strip("/").split("/")
    if len(parts) != 2 or not all(p and all(ch.isalnum() or ch in "-_." for ch in p) for p in parts):
        raise KaggleClientError(f"Dataset slug must look like 'owner/dataset-name', got {slug!r}")
    return "/".join(parts)


class KaggleClient:
    def __init__(self, api=None):
        self._api = api  # injectable for tests

    @property
    def api(self):
        if self._api is None:
            if credential_source() is None:
                raise KaggleAuthError(SETUP_HELP)
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi

                api = KaggleApi()
                api.authenticate()
            except (Exception, SystemExit) as exc:  # the kaggle client calls exit() on auth failure
                raise KaggleAuthError(f"{SETUP_HELP} (detail: {type(exc).__name__})") from None
            self._api = api
        return self._api

    def search(self, query: str, max_results: int = 20, sort_by: str = "votes", license_name: str | None = None) -> list[dict]:
        try:
            items = self.api.dataset_list(search=query, sort_by=sort_by, license_name=license_name) or []
        except KaggleAuthError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise KaggleClientError(f"Kaggle search failed: {exc}") from exc
        out = []
        for d in items[:max_results]:
            ref = getattr(d, "ref", None) or str(d)
            out.append({
                "ref": ref,
                "title": getattr(d, "title", ""),
                "size_bytes": getattr(d, "total_bytes", None) or getattr(d, "totalBytes", None),
                "license": getattr(d, "license_name", None) or getattr(d, "licenseName", None),
                "last_updated": str(getattr(d, "last_updated", "") or getattr(d, "lastUpdated", "")),
                "usability": getattr(d, "usability_rating", None) or getattr(d, "usabilityRating", None),
                "url": f"https://www.kaggle.com/datasets/{ref}",
            })
        return out

    def list_files(self, slug: str) -> list[dict]:
        slug = _safe_slug(slug)
        try:
            res = self.api.dataset_list_files(slug)
        except KaggleAuthError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise KaggleClientError(f"Could not list files for {slug}: {exc}") from exc
        files = getattr(res, "files", None) or getattr(res, "dataset_files", None) or []
        return [{"name": getattr(f, "name", str(f)), "size_bytes": getattr(f, "total_bytes", None)} for f in files]

    def download(self, slug: str, dest: str | Path | None = None, confirm: bool = False, unzip: bool = True,
                 license_name: str | None = None, annotation_format: str | None = None, notes: str = "") -> Path:
        """Download one explicitly selected dataset and record it in the dataset registry."""
        slug = _safe_slug(slug)
        if not confirm:
            raise KaggleClientError(
                f"Refusing to download '{slug}' without explicit confirmation. Re-run with --yes after checking "
                f"its size and license at https://www.kaggle.com/datasets/{slug}"
            )
        dest = Path(dest or PROJECT_ROOT / "data" / "raw" / slug.replace("/", "__"))
        dest.mkdir(parents=True, exist_ok=True)
        meta = {}
        try:
            self.api.dataset_metadata(slug, path=str(dest))
            mfile = dest / "dataset-metadata.json"
            if mfile.exists():
                import json

                meta = json.loads(mfile.read_text())
        except Exception as exc:  # noqa: BLE001 - metadata is nice-to-have
            log.warning("Could not fetch metadata for %s: %s", slug, exc)
        try:
            self.api.dataset_download_files(slug, path=str(dest), unzip=unzip, quiet=False)
        except KaggleAuthError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise KaggleClientError(f"Download of {slug} failed: {exc}") from exc
        licenses = meta.get("licenses") or []
        from src.data.dataset_registry import DatasetRegistry

        DatasetRegistry().register(
            name=meta.get("title") or slug,
            slug=slug,
            source_url=f"https://www.kaggle.com/datasets/{slug}",
            version=str(meta.get("versionNumber") or meta.get("currentVersionNumber") or "latest"),
            license=license_name or (licenses[0].get("name") if licenses else "UNKNOWN - verify on Kaggle page"),
            download_date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            annotation_format=annotation_format or "unknown",
            local_path=str(dest),
            notes=notes,
        )
        return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Search and download Kaggle datasets (explicit selection only).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=20)
    f = sub.add_parser("files")
    f.add_argument("slug")
    d = sub.add_parser("download")
    d.add_argument("slug")
    d.add_argument("--dest")
    d.add_argument("--format", dest="annotation_format")
    d.add_argument("--yes", action="store_true", help="confirm you checked the size and license")
    sub.add_parser("check-auth")
    args = ap.parse_args(argv)
    client = KaggleClient()
    try:
        if args.cmd == "check-auth":
            src = credential_source()
            print(f"Credential source: {src or 'none'}")
            client.api  # noqa: B018 - triggers authentication
            print("Kaggle authentication OK")
        elif args.cmd == "search":
            for r in client.search(args.query, args.n):
                size = f"{r['size_bytes'] / 1e9:.2f} GB" if r["size_bytes"] else "?"
                print(f"{r['ref']:<55} {size:>9}  license={r['license']}  {r['title']}")
        elif args.cmd == "files":
            for r in client.list_files(args.slug):
                print(r["name"], r["size_bytes"])
        elif args.cmd == "download":
            print(client.download(args.slug, args.dest, confirm=args.yes, annotation_format=args.annotation_format))
    except (KaggleAuthError, KaggleClientError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
