"""The public test set, downloaded from Google Drive on first use and verified before it is used.

The folder holds the speaker-disjoint test split of our Vaani-Hi rebuild, as written by scripts/export_testset.py:
testset.parquet (audio and labels), transcripts_w2v2.parquet, transcripts_zipa.parquet, README.txt and SHA256SUMS.

`ensure()` returns data/testset/ once the three data files match the SHA-256 digests below, which are fixed here.
README.txt and SHA256SUMS are not checked, so editing them on Drive breaks nothing. A download goes to
data/testset.download/ first, resumes after an interruption, fetches files that fail the check once more, and is moved
into place only when every data file passes, so data/testset/ never holds an unverified data file. Concurrent callers
wait on a lock.

The digests pin the files uploaded on 26 Nov 2026. Do not replace the data files on Drive; if they must change, update
SHA256 here in the same commit (scripts/export_testset.py prints whether a fresh export still matches).
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from sal.paths import DATA

URL = "https://drive.google.com/drive/folders/16y49znRs_ojTTsIt91AJ0aLzDm8geLSi"
DIR = DATA / "testset"
SHA256 = {
    "testset.parquet": "492220edbb02d347eb835d371da35ef3bf40e87bae117d0ebf518f40d55dc855",
    "transcripts_w2v2.parquet": "8fdf3062aee6b0aaeccdd1a7cbf405e8a93b25e6cec92cbab1a97e607fa4bb56",
    "transcripts_zipa.parquet": "9564b2c223c20f212c82ce4bf944109304af60b256cad0bf9afd27d179818ffc",
}
RECOGNIZERS = ("w2v2", "zipa")


class TestSetError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def failures(folder: Path) -> list[str]:
    """Data files of the test set that are missing from `folder` or differ from the expected digests."""
    return [name for name, digest in SHA256.items()
            if not (folder / name).is_file() or sha256(folder / name) != digest]


def _download(folder: Path, tries: int = 4) -> None:
    """Fetch the Drive folder into `folder`. A dropped connection leaves a partial file that the next try resumes."""
    import time

    import gdown

    for k in range(1, tries + 1):
        try:
            gdown.download_folder(url=URL, output=str(folder), quiet=False, use_cookies=False, resume=True)
            return
        except Exception as e:  # gdown raises several types: network errors, Drive quota pages, changed folder layout
            if k == tries:
                raise TestSetError(f"could not download the test set from {URL} after {tries} tries "
                                   f"({type(e).__name__}: {e}). Download the folder by hand and put its files in "
                                   f"{DIR}") from e
            print(f"download interrupted ({type(e).__name__}); resuming, try {k + 1} of {tries}")
            time.sleep(5 * k)


def ensure(folder: Path = DIR) -> Path:
    """Return `folder` holding the verified test set, downloading it first if needed."""
    if not failures(folder):
        return folder
    from filelock import FileLock

    folder.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder) + ".lock"):
        if not failures(folder):           # another process finished the download while we waited
            return folder
        tmp = folder.with_name(folder.name + ".download")
        tmp.mkdir(exist_ok=True)
        if folder.is_dir():                # keep the files that already pass, so only the rest is fetched
            for name, digest in SHA256.items():
                if (folder / name).is_file() and not (tmp / name).exists() and sha256(folder / name) == digest:
                    shutil.copy2(folder / name, tmp / name)
        print(f"downloading the test set (about 700 MB) to {folder}")
        for attempt in (1, 2):
            for name in failures(tmp):     # gdown skips existing files, so remove the ones that are wrong
                (tmp / name).unlink(missing_ok=True)
            _download(tmp)
            bad = failures(tmp)
            if not bad:
                break
            print(f"attempt {attempt}: {bad} failed the checksum")
        else:
            raise TestSetError(f"after two downloads these files still fail the checksum: {bad}. Delete {tmp} and "
                               f"retry, or download {URL} by hand into {folder}")
        if folder.exists():
            shutil.rmtree(folder)
        tmp.rename(folder)
    return folder


def transcripts(recognizer: str, folder: Path | None = None) -> Path:
    """The test set's transcripts for `recognizer` (w2v2 or zipa), downloading the test set if needed."""
    if recognizer not in RECOGNIZERS:
        raise TestSetError(f"the test set has transcripts for {RECOGNIZERS}, not {recognizer!r}")
    return ensure(folder or DIR) / f"transcripts_{recognizer}.parquet"
