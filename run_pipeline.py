import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from config import ensure_dirs

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINE_STEPS = [
    {
        "name": "download_prices",
        "cmd": [
            sys.executable,
            "-c",
            "from src.downloadPrices import DownloadPrices; DownloadPrices(verbose=True).main()",
        ],
    },
    {
        "name": "prepare_weekly_adjclose",
        "cmd": [sys.executable, "src/02_prepare_weekly_adjclose.py"],
    },
    {
        "name": "filter_coverage",
        "cmd": [sys.executable, "src/03_filter_coverage.py"],
    },
    {
        "name": "make_weekly_returns",
        "cmd": [sys.executable, "src/04_make_weekly_returns.py"],
    },
    {
        "name": "freeze_universe",
        "cmd": [sys.executable, "src/06_freeze_universe.py"],
    },
    {
        "name": "build_x_only",
        "cmd": [sys.executable, "-m", "src.07_build_X_only"],
    },
    {
        "name": "build_snapshots",
        "cmd": [sys.executable, "-m", "src.05_build_snapshots"],
    },
    {
        "name": "build_x_raw_snapshots",
        "cmd": [sys.executable, "-m", "src.09_build_X_raw_snapshots"],
    },
    {
        "name": "build_embeddings_gat",
        "cmd": [sys.executable, "-m", "src.block3.embed"],
    },
]

REQUIRED_ENV_VARS = [
    "TICKERS_URL",
    "TICKERS_DIR",
    "START_DATE",
    "INTERVAL",
    "DATA_PROCESSED_DIR",
    "PRICES_WEEKLY_FILE",
]


def validate_env() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Faltan variables de entorno requeridas: {missing}. "
            f"Configúralas en tu shell o en un archivo .env"
        )


def run_step(step: dict):
    print(f"\n[INFO] Ejecutando: {step['name']}")
    print(f"[INFO] Command: {' '.join(step['cmd'])}")

    env = os.environ.copy()

    result = subprocess.run(
        step["cmd"],
        cwd=PROJECT_ROOT,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Falló el paso '{step['name']}' con código {result.returncode}"
        )


def main():
    ensure_dirs()
    validate_env()

    for step in PIPELINE_STEPS:
        run_step(step)

    print("\n[OK] Pipeline completo ejecutado con éxito")


if __name__ == "__main__":
    main()