import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    cmd = [sys.executable, "-m", "src.10_linucb_contextual"]

    print("[INFO] Ejecutando bandits")
    print(f"[INFO] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(f"Falló 10_linucb_contextual con código {result.returncode}")

    print("[OK] Bandits ejecutados correctamente")

if __name__ == "__main__":
    main()