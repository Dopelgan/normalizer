"""Временные файлы процесса: каталог в томе сервиса, а не на слое контейнера."""

import os
import tempfile

import pytest

from core.config import settings
from core import workspace


@pytest.fixture(autouse=True)
def _restore_process_tempdir():
    """Тест меняет глобальное состояние процесса — возвращаем как было."""
    saved_env = {name: os.environ.get(name) for name in ("TMPDIR", "TEMP", "TMP")}
    saved_tempdir = tempfile.tempdir
    yield
    tempfile.tempdir = saved_tempdir
    for name, value in saved_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class TestTempDir:
    def test_creates_directory(self, tmp_path, monkeypatch):
        target = tmp_path / "shared" / "tmp"
        monkeypatch.setattr(settings, "TEMP_DIR", str(target))
        assert workspace.temp_dir() == target
        assert target.is_dir()

    def test_fallback_is_system_tempdir(self, tmp_path, monkeypatch):
        """Том не смонтирован — сервис не падает, уходит в системный каталог."""
        blocker = tmp_path / "file"
        blocker.write_bytes(b"x")
        monkeypatch.setattr(settings, "TEMP_DIR", str(blocker / "tmp"))
        assert str(workspace.temp_dir()) == tempfile.gettempdir()


class TestConfigureProcessTempdir:
    def test_sets_env_and_tempfile_default(self, tmp_path, monkeypatch):
        target = tmp_path / "vol" / "tmp"
        monkeypatch.setattr(settings, "TEMP_DIR", str(target))

        workspace.configure_process_tempdir()

        assert tempfile.tempdir == str(target)
        for name in ("TMPDIR", "TEMP", "TMP"):
            assert os.environ[name] == str(target)

    def test_new_temp_files_land_in_volume(self, tmp_path, monkeypatch):
        target = tmp_path / "vol" / "tmp"
        monkeypatch.setattr(settings, "TEMP_DIR", str(target))
        workspace.configure_process_tempdir()

        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            assert handle.name.startswith(str(target))
