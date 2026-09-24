#!/usr/bin/env python3
"""Copy the exact experiment source trees without data, results, or weights.

This helper is meant to run next to the original ICASSP_2027 projects on
Huairou. It never writes to a source project and refuses to replace a changed
file in Code_Release unless --force is specified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import shutil


RELEASE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RELEASE.parent
PROJECTS = (
    "VAD_pkueeg_final",
    "VAD_SparKULee",
    "VAD_MEG_NIPS",
    "VAD_SEM4Lang",
    "VAD_pkueeg_scaling",
    "VAD_SParKULee_scaling",
    "VAD_MEG_NIPS_scaling",
    "VAD_SEM4Lang_scaling",
)
CODE_SUFFIXES = {".py", ".sh", ".slurm", ".yaml", ".yml", ".md", ".txt", ".toml", ".ini"}
ROOT_CODE_DIRS = ("speech_code", "configs", "tests", "scaling", "preprocess")
EXCLUDED_PARTS = {
    "__pycache__",
    ".git",
    ".integration_checks",
    "code_backups",
    "data",
    "datasets",
    "results",
    "log",
    "logs",
    "output",
    "outputs",
    "checkpoints",
    "weights",
    "models_cache",
}
EXTRA_FILES = {
    "VAD_pkueeg_final": (
        ("Label_VAD/extract_vad_segments.py", "../Label_VAD/extract_vad_segments.py"),
    ),
    "VAD_SparKULee": (
        ("Label_VAD/extract_sparkulee_vad_segments.py", "Label_VAD/extract_sparkulee_vad_segments.py"),
        ("SparKULee/preprocess/extract_eeg_250hz_bp.py", "SparKULee/preprocess/extract_eeg_250hz_bp.py"),
        ("SparKULee/preprocess/job_extract.slurm", "SparKULee/preprocess/job_extract.slurm"),
    ),
    "VAD_MEG_NIPS": (("ch_types.npz", "ch_types.npz"),),
    "VAD_SEM4Lang": (
        ("Label_VAD/extract_vad_segments.py", "Label_VAD/extract_vad_segments.py"),
        ("Label_VAD/plot_envelope_with_vad.py", "Label_VAD/plot_envelope_with_vad.py"),
        ("Label_VAD/job_extract_vad.slurm", "Label_VAD/job_extract_vad.slurm"),
        ("Label_VAD/env/extract_envelope.py", "Label_VAD/env/extract_envelope.py"),
        ("SEM4Lang/preprocess_meg.py", "SEM4Lang/preprocess_meg.py"),
        ("SEM4Lang/job_preprocess_meg.slurm", "SEM4Lang/job_preprocess_meg.slurm"),
    ),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_files(project: str):
    root = SOURCE_ROOT / project
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in CODE_SUFFIXES:
            yield path, Path(project) / path.name
    for directory in ROOT_CODE_DIRS:
        source_dir = root / directory
        if not source_dir.is_dir():
            continue
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
                continue
            if any(part in EXCLUDED_PARTS or part.startswith(".") for part in path.relative_to(source_dir).parts):
                continue
            yield path, Path(project) / path.relative_to(root)
    for destination, source in EXTRA_FILES.get(project, ()):
        src = root / source
        if not src.is_file():
            raise FileNotFoundError(src)
        yield src, Path(project) / destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace differing release copies")
    args = parser.parse_args()

    rows = []
    seen = set()
    for project in PROJECTS:
        count = 0
        for source, relative in source_files(project):
            if relative in seen:
                raise RuntimeError(f"duplicate destination: {relative}")
            seen.add(relative)
            if source.stat().st_size > 1_000_000:
                raise RuntimeError(f"unexpected large code file: {source}")
            target = RELEASE / relative
            sha = digest(source)
            if target.exists() and digest(target) != sha and not args.force:
                raise RuntimeError(f"changed release file; refusing to overwrite: {target}")
            if not args.dry_run and (not target.exists() or digest(target) != sha):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            rows.append((relative.as_posix(), source.relative_to(SOURCE_ROOT).as_posix(), source.stat().st_size, sha))
            count += 1
        print(f"{project}: {count} source files")

    if not args.dry_run:
        manifest = RELEASE / "SOURCE_MANIFEST.csv"
        with manifest.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("release_path", "source_path", "bytes", "sha256"))
            writer.writerows(rows)
    print(f"Total: {len(rows)} files, {sum(row[2] for row in rows)} bytes")


if __name__ == "__main__":
    main()
