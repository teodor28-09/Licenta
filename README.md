# CARMA – workspace local de învățare

## 1. Structura
```
carma_workspace/
  CARMA/                         <- git clone https://github.com/adrian-naziru/CARMA
    data/nuscenes/v1.0-mini/     <- nuScenes mini (nuscenes.org, ~4 GB)
    checkpoints/CARMA_map0.4922.pth   <- de pe huggingface.co/AdrianNaziru/CARMA
  scripts/
  outputs/                       <- imaginile generate
```

## 2. Mediul (o singură dată)
```bash
cd carma_workspace
git clone https://github.com/adrian-naziru/CARMA
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/Mac: source .venv/bin/activate
pip install torch torchvision          # cu CUDA: vezi comanda de pe pytorch.org
pip install nuscenes-devkit opencv-python matplotlib pyquaternion pillow numpy
```

## 3. Rulare (din folderul carma_workspace)
```bash
python scripts/01_explore_nuscenes.py --index 0                 # date brute
python scripts/02_dataset_targets.py  --split val --index 0     # input + target CARMA
python scripts/03_forward_hooks.py    --split val --index 0     # modelul, etapă cu etapă
python scripts/03_forward_hooks.py    --split val --index 0 --zero-radar
python scripts/03_forward_hooks.py    --split val --index 0 --zero-camera
python scripts/03_forward_hooks.py    --split val --index 0 --score 0.2
```
`mini_val` are 2 scene (~80 sample-uri), ambele din split-ul oficial de validare,
deci modelul nu le-a văzut la antrenare.

Altă locație pentru date/checkpoint: setează `NUSCENES_DATAROOT`, `CARMA_CKPT`, `CARMA_DIR`.
