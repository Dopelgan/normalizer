"""Общие фикстуры тестов."""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db.models import Base                     # noqa: E402
from core.models.parse_result import ParsedBlock    # noqa: E402
from core.providers.storage import LocalStorageProvider  # noqa: E402

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


# ------------------------------------------------------------------ БД
@pytest.fixture
def db_session():
    """Сессия SQLite в памяти со свежей схемой."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# -------------------------------------------------------------- Redis
@pytest.fixture
def fake_redis():
    """Подменяет клиент агрегатора на fakeredis."""
    fakeredis = pytest.importorskip("fakeredis")
    from core import result_aggregator

    client = fakeredis.FakeRedis(decode_responses=True)
    result_aggregator.set_client(client)
    try:
        yield client
    finally:
        result_aggregator.set_client(None)


# ---------------------------------------------------------- Хранилище
@pytest.fixture
def temp_storage(tmp_path):
    return LocalStorageProvider(base_path=str(tmp_path))


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def sample_pdf_path():
    return FIXTURES_DIR / "sample.pdf"


# ------------------------------------------------------------- Блоки
def make_block(text, page=1, bbox=None, order=None, **kwargs):
    return ParsedBlock(
        type=kwargs.pop("type", "text"),
        text=text,
        page=page,
        bbox=bbox or [0.1, 0.1, 0.9, 0.3],
        order=order,
        **kwargs,
    )


@pytest.fixture
def text_blocks():
    """Страница с заголовком и тремя абзацами."""
    return [
        make_block("1. Общие положения", page=1, bbox=[0.1, 0.05, 0.9, 0.1], order=0),
        make_block("Настоящий документ устанавливает требования к изделию. " * 3,
                   page=1, bbox=[0.1, 0.12, 0.9, 0.25], order=1),
        make_block("2. Технические требования", page=1, bbox=[0.1, 0.3, 0.9, 0.35], order=2),
        make_block("Шероховатость поверхности не более Ra 3.2 мкм. " * 3,
                   page=1, bbox=[0.1, 0.37, 0.9, 0.5], order=3),
        make_block("Материал детали — сталь 45 ГОСТ 1050-2013. " * 3,
                   page=2, bbox=[0.1, 0.1, 0.9, 0.25], order=4),
    ]
