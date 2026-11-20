import io
import tarfile
import zipfile

import pytest

from cvdoc.datasets.manifest import extract_archive


@pytest.mark.parametrize("name", ["../evil", "/absolute", "C:/escape", "..\\evil"])
def test_zip_traversal(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, "bad")
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out")


def test_tar_symlink(tmp_path):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside"
        archive.addfile(info)
    with pytest.raises(ValueError):
        extract_archive(path, tmp_path / "out")


def test_normal_tar(tmp_path):
    path = tmp_path / "good.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("folder/file.txt")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"ok"))
    assert extract_archive(path, tmp_path / "out") == 1
    assert (tmp_path / "out/folder/file.txt").read_text() == "ok"
