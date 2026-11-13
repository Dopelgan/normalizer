"""
Выбор каталога моделей Tesseract.

Подмена модели — это чтение документов заказчика другим файлом, и ошибка
здесь должна быть видна, а не тиха. Проверяется ровно это: по умолчанию
каталог системный; выбранный каталог подставляется целиком; неполный или
отсутствующий каталог не роняет распознавание, а возвращает к системной
модели с предупреждением в логе.
"""

import logging
import shlex

from core.config import settings
from core.providers import ocr_model


def _tessdata(tmp_path, names=ocr_model.REQUIRED):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in names:
        # Содержимое не важно: проверяется только наличие файлов.
        (tmp_path / name).write_bytes(b"traineddata")
    return str(tmp_path)


class TestTessdataDir:
    def test_empty_setting_means_system_model(self, monkeypatch):
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", "")
        assert ocr_model.tessdata_dir() is None
        assert ocr_model.tessdata_option() == ""

    def test_complete_directory_is_used(self, monkeypatch, tmp_path):
        directory = _tessdata(tmp_path)
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", directory)
        assert ocr_model.tessdata_dir() == directory
        assert shlex.split(ocr_model.tessdata_option()) == ["--tessdata-dir", directory]

    def test_surrounding_spaces_do_not_break_the_path(self, monkeypatch, tmp_path):
        directory = _tessdata(tmp_path)
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", f"  {directory}  ")
        assert ocr_model.tessdata_dir() == directory

    def test_missing_directory_falls_back_to_system(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(ocr_model, "_warned", set())
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", str(tmp_path / "нет-такого"))
        with caplog.at_level(logging.WARNING):
            assert ocr_model.tessdata_dir() is None
        assert "OCR_TESSDATA_DIR" in caplog.text

    def test_incomplete_directory_falls_back_to_system(self, monkeypatch, tmp_path, caplog):
        # `--tessdata-dir` подменяет каталог целиком: без eng не соберётся
        # связка rus+eng. Половина каталога хуже системного — её не берём.
        directory = _tessdata(tmp_path, ("rus.traineddata",))
        monkeypatch.setattr(ocr_model, "_warned", set())
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", directory)
        with caplog.at_level(logging.WARNING):
            assert ocr_model.tessdata_dir() is None
        assert "eng.traineddata" in caplog.text

    def test_path_with_a_space_is_quoted(self, monkeypatch, tmp_path):
        # pytesseract разбирает строку конфигурации через shlex.split:
        # неэкранированный пробел развалил бы вызов на два слова.
        directory = _tessdata(tmp_path / "модели best", )
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", directory)
        option = ocr_model.tessdata_option()
        assert shlex.split(option) == ["--tessdata-dir", directory]

    def test_warning_is_logged_once_per_directory(self, monkeypatch, tmp_path, caplog):
        # Лист за листом один и тот же каталог не должен забивать лог.
        monkeypatch.setattr(ocr_model, "_warned", set())
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", str(tmp_path / "нет-такого"))
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                ocr_model.tessdata_dir()
        assert caplog.text.count("OCR_TESSDATA_DIR") == 1
