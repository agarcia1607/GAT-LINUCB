from pathlib import Path
import os

# ---------------------------------------------------
# Paths del proyecto
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# carpeta donde están tus scripts actuales
SRC_DIR = PROJECT_ROOT / "src"

# ---------------------------------------------------
# Directorios de datos
# ---------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
GRAPHS_DIR = DATA_DIR / "graphs"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
BANDITS_DIR = DATA_DIR / "bandits"
LOGS_DIR = DATA_DIR / "logs"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MANIFESTS_DIR = OUTPUTS_DIR / "manifests"

# ---------------------------------------------------
# Parámetros del experimento
# ---------------------------------------------------

SEED = int(os.getenv("SEED", "42"))

# ---------------------------------------------------
# Configuración AWS (para más adelante)
# ---------------------------------------------------

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "portfolio-bandit-project")

# ---------------------------------------------------
# Crear directorios si no existen
# ---------------------------------------------------

def ensure_dirs():
    for p in [
        DATA_DIR,
        RAW_DIR,
        FEATURES_DIR,
        GRAPHS_DIR,
        SNAPSHOTS_DIR,
        BANDITS_DIR,
        LOGS_DIR,
        OUTPUTS_DIR,
        MANIFESTS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)