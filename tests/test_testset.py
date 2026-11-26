"""sal.testset: verification and download logic, with the Drive download replaced by a local fake."""
import hashlib
from pathlib import Path

import pytest

from sal import testset

FILES = {"testset.parquet": b"audio and labels", "transcripts_w2v2.parquet": b"w2v2",
         "transcripts_zipa.parquet": b"zipa"}


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


SUMS = "".join(f"{digest(b)}  {n}\n" for n, b in FILES.items())


@pytest.fixture(autouse=True)
def fake_digests(monkeypatch):
    monkeypatch.setattr(testset, "SHA256", {n: digest(b) for n, b in FILES.items()})


def write(folder, corrupt=(), skip=()):
    folder.mkdir(parents=True, exist_ok=True)
    for n, b in FILES.items():
        if n not in skip:
            (folder / n).write_bytes(b"corrupt" if n in corrupt else b)
    (folder / "SHA256SUMS").write_text(SUMS)
    (folder / "README.txt").write_text("readme")


class FakeDrive:
    """Stands in for gdown: like gdown, it leaves files that already exist alone."""

    def __init__(self, corrupt_rounds=()):
        self.calls, self.corrupt_rounds, self.missing_at_call = 0, list(corrupt_rounds), []

    def __call__(self, folder):
        corrupt = self.corrupt_rounds[self.calls] if self.calls < len(self.corrupt_rounds) else ()
        self.calls += 1
        self.missing_at_call.append(sorted(n for n in FILES if not (folder / n).exists()))
        for n, b in FILES.items():
            if not (folder / n).exists():
                (folder / n).write_bytes(b"corrupt" if n in corrupt else b)
        for extra, text in (("SHA256SUMS", SUMS), ("README.txt", "readme")):
            if not (folder / extra).exists():
                (folder / extra).write_text(text)


def test_verified_folder_is_used_without_downloading(tmp_path, monkeypatch):
    write(tmp_path / "t")
    drive = FakeDrive()
    monkeypatch.setattr(testset, "_download", drive)
    assert testset.ensure(tmp_path / "t") == tmp_path / "t"
    assert drive.calls == 0


def test_missing_folder_is_downloaded_verified_and_moved_into_place(tmp_path, monkeypatch):
    drive = FakeDrive()
    monkeypatch.setattr(testset, "_download", drive)
    out = testset.ensure(tmp_path / "t")
    assert drive.calls == 1 and testset.failures(out) == []
    assert not (tmp_path / "t.download").exists()


def test_corrupt_download_is_fetched_again(tmp_path, monkeypatch):
    drive = FakeDrive(corrupt_rounds=[("testset.parquet",)])
    monkeypatch.setattr(testset, "_download", drive)
    out = testset.ensure(tmp_path / "t")
    assert drive.calls == 2 and testset.failures(out) == []
    assert drive.missing_at_call[1] == ["testset.parquet"]       # only the bad file was fetched again


def test_persistent_corruption_raises_and_leaves_no_unverified_folder(tmp_path, monkeypatch):
    drive = FakeDrive(corrupt_rounds=[("transcripts_w2v2.parquet",), ("transcripts_w2v2.parquet",)])
    monkeypatch.setattr(testset, "_download", drive)
    with pytest.raises(testset.TestSetError, match="transcripts_w2v2"):
        testset.ensure(tmp_path / "t")
    assert not (tmp_path / "t").exists()


def test_partly_valid_folder_keeps_good_files_and_replaces_bad_ones(tmp_path, monkeypatch):
    write(tmp_path / "t", corrupt=("transcripts_zipa.parquet",), skip=("transcripts_w2v2.parquet",))
    drive = FakeDrive()
    monkeypatch.setattr(testset, "_download", drive)
    out = testset.ensure(tmp_path / "t")
    assert testset.failures(out) == []
    assert drive.missing_at_call[0] == ["transcripts_w2v2.parquet", "transcripts_zipa.parquet"]


def test_edited_readme_or_sums_on_drive_do_not_matter(tmp_path, monkeypatch):
    write(tmp_path / "t")
    (tmp_path / "t" / "README.txt").write_text("edited after upload")
    (tmp_path / "t" / "SHA256SUMS").unlink()
    drive = FakeDrive()
    monkeypatch.setattr(testset, "_download", drive)
    assert testset.ensure(tmp_path / "t") == tmp_path / "t" and drive.calls == 0


def test_download_errors_become_a_clear_message(tmp_path, monkeypatch):
    import time

    import gdown

    calls = []

    def refuse(**kw):
        calls.append(1)
        raise RuntimeError("Too many users have viewed or downloaded this file recently")
    monkeypatch.setattr(gdown, "download_folder", refuse)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(testset.TestSetError, match="by hand"):
        testset.ensure(tmp_path / "t")
    assert len(calls) == 4                                   # retried before giving up


def test_interrupted_download_is_resumed(tmp_path, monkeypatch):
    import time

    import gdown

    calls = []

    def flaky(output, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("connection reset")
        FakeDrive()(Path(output))
    monkeypatch.setattr(gdown, "download_folder", flaky)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert testset.failures(testset.ensure(tmp_path / "t")) == [] and len(calls) == 2


def test_unknown_recognizer_is_refused(tmp_path):
    with pytest.raises(testset.TestSetError, match="whisper"):
        testset.transcripts("whisper", tmp_path / "t")
