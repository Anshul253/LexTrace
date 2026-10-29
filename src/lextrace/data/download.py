"""Dataset acquisition from the public Hugging Face mirror `awinml/legalbench-rag`, pinned to one revision.

Only `benchmarks/cuad.json` and the contracts referenced by the 194 mini queries are downloaded. Each file is
checked against the mirror's `SHA256SUMS` when that file lists it. A manifest with hashes is written to
`data/dataset_manifest.json` (committed) so later runs can detect a changed dataset."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

from ..paths import BENCHMARK_FILE, CORPUS_DIR, MANIFEST_FILE, RAW_DIR
from .benchmark import load_mini_queries, referenced_documents
from .corpus import locate

REPO_ID = "awinml/legalbench-rag"
REVISION = "2b9c248bc8179ef0908fd6ba01d50b156facd48b"


class DownloadError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(rel: str, revision: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise DownloadError("huggingface_hub is not installed: pip install -r requirements.txt") from e
    last: Optional[Exception] = None
    for cand in dict.fromkeys([rel, unicodedata.normalize("NFC", rel), unicodedata.normalize("NFD", rel)]):
        try:
            return Path(hf_hub_download(REPO_ID, cand, repo_type="dataset", revision=revision, local_dir=RAW_DIR))
        except Exception as e:  # noqa: BLE001 - try the next spelling, report the last error
            last = e
    raise DownloadError(
        f"could not download {rel!r} from {REPO_ID}@{revision[:8]}: {last}\n"
        "Recovery: check network access to huggingface.co (or set HF_ENDPOINT to a mirror), then re-run "
        "`python -m lextrace prepare`. Files can also be fetched manually from "
        f"https://huggingface.co/datasets/{REPO_ID}/tree/{revision} into {RAW_DIR}.")


def _checksums(revision: str) -> Dict[str, str]:
    try:
        path = _fetch("SHA256SUMS", revision)
    except DownloadError:
        return {}
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[unicodedata.normalize("NFC", name.strip().lstrip("*").lstrip("./"))] = digest
    return sums


def prepare(revision: str = REVISION, log: Callable[[str], None] = print) -> Dict:
    """Download (idempotently) and write the manifest. Returns the manifest dict."""
    sums = _checksums(revision)
    log(f"checksum list: {len(sums)} entries" if sums else "checksum list unavailable; recording hashes only")
    if not BENCHMARK_FILE.exists():
        _fetch("benchmarks/cuad.json", revision)
    files = {"benchmarks/cuad.json": sha256_file(BENCHMARK_FILE)}
    queries = load_mini_queries()
    docs = referenced_documents(queries)
    log(f"mini selection: {len(queries)} queries over {len(docs)} contracts")
    for i, d in enumerate(docs, 1):
        path = locate(CORPUS_DIR, d) or _fetch(f"corpus/{d}", revision)
        files[f"corpus/{d}"] = sha256_file(path)
        if i % 10 == 0:
            log(f"  {i}/{len(docs)} contracts ready")
    mismatches = [k for k, v in files.items() if unicodedata.normalize("NFC", k) in sums
                  and sums[unicodedata.normalize("NFC", k)] != v]
    if mismatches:
        raise DownloadError(f"checksum mismatch for {mismatches[:3]}; delete {RAW_DIR} and re-run prepare")
    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "revision": revision,
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_queries": len(queries),
        "n_documents": len(docs),
        "checksums_verified": sum(1 for k in files if unicodedata.normalize("NFC", k) in sums),
        "files": files,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {MANIFEST_FILE}")
    return manifest
