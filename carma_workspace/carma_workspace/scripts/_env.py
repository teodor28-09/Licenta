"""
Setari comune pentru toate scripturile din workspace.

Structura asteptata:
    carma_workspace/
        CARMA/                      <- git clone https://github.com/adrian-naziru/CARMA
            data/nuscenes/v1.0-mini/
            checkpoints/CARMA_map0.4922.pth
        scripts/                    <- scripturile astea
        outputs/                    <- aici se salveaza imaginile

Poti suprascrie caile cu variabile de mediu:
    CARMA_DIR, NUSCENES_DATAROOT, NUSCENES_VERSION, CARMA_CKPT
"""
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
CARMA_DIR = Path(os.environ.get("CARMA_DIR", WORKSPACE / "CARMA")).resolve()
VERSION = os.environ.get("NUSCENES_VERSION", "v1.0-mini")
DATA_ROOT = os.environ.get("NUSCENES_DATAROOT", str(CARMA_DIR / "data" / "nuscenes" / VERSION))
CHECKPOINT = os.environ.get("CARMA_CKPT", str(CARMA_DIR / "checkpoints" / "CARMA_map0.4922.pth"))
OUT_DIR = WORKSPACE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup(profile: str = "trainval", amp: bool = False) -> None:
    """Aplica profilul de antrenare CARMA in variabilele de mediu.

    IMPORTANT: trebuie apelat INAINTE de `import config` / `import model_*`,
    pentru ca dimensiunile straturilor se citesc din env la import (vezi teza, 3.8).
    """
    if not CARMA_DIR.is_dir():
        raise SystemExit(f"Nu gasesc repo-ul CARMA la {CARMA_DIR}. Seteaza CARMA_DIR.")
    sys.path.insert(0, str(CARMA_DIR))
    os.chdir(CARMA_DIR)

    import run_train_module  # defineste doar PROFILES, nu importa config

    for k, v in run_train_module.PROFILES[profile].items():
        os.environ[str(k)] = str(v)

    # suprascrieri pentru lucru local, un sample odata
    os.environ.update({
        "NUSCENES_DATAROOT": DATA_ROOT,
        "NUSCENES_VERSION": VERSION,
        "TRAIN_STAGE": "all",
        "BATCH_SIZE": "1",
        "NUM_WORKERS": "0",
        "USE_CBGS": "0",
        "USE_GRADIENT_CHECKPOINTING": "0",
        "AMP": "1" if amp else "0",
    })
    print(f"[env] profil={profile} | versiune={VERSION} | date={DATA_ROOT}")
