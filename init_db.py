"""
Инициализация схемы БД.

Схема ведётся Alembic'ом. Скрипт прогоняет миграции до head, а если Alembic
по какой-то причине недоступен — создаёт таблицы напрямую, чтобы поднять
окружение для разработки.
"""

import logging
import sys

from sqlalchemy import create_engine

from core.config import settings
from core.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("init_db")


def run_migrations() -> bool:
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        logger.warning("Alembic не установлен")
        return False

    try:
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
        command.upgrade(config, "head")
        logger.info("Миграции применены до head")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось применить миграции (%s), создаём таблицы напрямую", exc)
        return False


def create_all() -> None:
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(engine)
    logger.info("Таблицы созданы (если их не было)")


def init_db() -> None:
    if not run_migrations():
        create_all()


if __name__ == "__main__":
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001
        logger.error("Инициализация БД не удалась: %s", exc)
        sys.exit(1)
