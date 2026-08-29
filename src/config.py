import os
import configparser
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def config_path() -> Path:
    """Путь к config.ini."""
    return Path(os.getenv("CONFIG_PATH", ROOT / "config.ini"))


def load_config(path: Path = None) -> configparser.ConfigParser:
    """Загрузка конфига."""
    path = Path(path) if path else config_path()

    if not path.exists():
        raise FileNotFoundError(
            f"config.ini не найден: {path}"
        )

    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")

    return cfg


def resolve(rel_path: Path) -> Path:
    """Относительный путь из конфига -> абсолютный от корня репозитория."""
    p = Path(rel_path)

    return p if p.is_absolute() else ROOT / p


def model_path(
    cfg: configparser.ConfigParser = None
) -> Path:
    """
    Путь к локальной модели.

    MODEL_PATH из окружения имеет приоритет над config.ini.
    """
    env = os.getenv("MODEL_PATH")

    if env:
        path = Path(env)
    else:
        cfg = cfg or load_config()

        raw_path = cfg["MODEL"].get("model_path", "").strip()

        if not raw_path:
            raise ValueError(
                "MODEL.model_path не указан в config.ini"
            )

        path = resolve(raw_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Модель не найдена по пути: {path}"
        )

    if not path.is_dir():
        raise ValueError(
            f"model_path должен указывать на директорию модели: {path}"
        )

    return path


def images_path(
    cfg: configparser.ConfigParser = None
) -> Path:
    """Путь к изображениям."""
    env = os.getenv("IMAGES_PATH")

    if env:
        return Path(env)

    cfg = cfg or load_config()

    return resolve(cfg["DATA"]["images_path"])


def target_path(
    cfg: configparser.ConfigParser = None
) -> Path:
    """Путь к CSV с таргетами."""
    env = os.getenv("TARGET_PATH")

    if env:
        return Path(env)

    cfg = cfg or load_config()

    return resolve(cfg["DATA"]["csv_path"])