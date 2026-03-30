import os
import subprocess
import sys
import glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
LINUCB_ROOT = PROJECT_ROOT / "artifacts" / "linucb"


def get_latest_run_dir() -> Path:
    candidates = [p for p in LINUCB_ROOT.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found in {LINUCB_ROOT}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_bandits() -> Path:
    before = {p.resolve() for p in LINUCB_ROOT.iterdir() if p.is_dir()} if LINUCB_ROOT.exists() else set()

    cmd = [sys.executable, "-m", "src.10_linucb_contextual"]
    print("\n[INFO] Ejecutando bandits")
    print(f"[INFO] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Falló 10_linucb_contextual con código {result.returncode}")

    print("[OK] Bandits ejecutados correctamente")

    after = {p.resolve() for p in LINUCB_ROOT.iterdir() if p.is_dir()} if LINUCB_ROOT.exists() else set()
    new_dirs = list(after - before)

    if len(new_dirs) == 1:
        run_dir = Path(new_dirs[0])
        print(f"[INFO] Nueva corrida detectada: {run_dir}")
        return run_dir
    if len(new_dirs) > 1:
        latest = max((Path(p) for p in new_dirs), key=lambda p: p.stat().st_mtime)
        print(f"[WARN] Múltiples corridas nuevas — usando la más reciente: {latest}")
        return latest

    print("[WARN] No se detectó nueva carpeta — usando la más reciente")
    return get_latest_run_dir()


def upload_results_to_s3(run_dir: Path) -> None:
    from src.storage.s3_io import upload_dir

    bucket = os.getenv("S3_BUCKET")
    base_prefix = os.getenv("S3_BASE_PREFIX", "runs")
    run_id = os.getenv("RUN_ID", "run_local")

    if not bucket:
        print("[INFO] S3_BUCKET no definido — omitiendo upload a S3")
        return

    prefix = f"{base_prefix}/{run_id}/bandits/{run_dir.name}"
    print(f"\n[INFO] Subiendo resultados a s3://{bucket}/{prefix}/")
    upload_dir(run_dir, bucket, prefix)
    print("[OK] Resultados subidos a S3")


def main() -> None:
    run_dir = run_bandits()
    upload_results_to_s3(run_dir)


if __name__ == "__main__":
    main()