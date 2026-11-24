"""
Pasul 2 - ce primeste CARMA si ce trebuie sa invete (input + target).

Ce vezi: shape-urile tensorilor din NuScenesDataset, heatmap-ul tinta
(petele gaussiene din centrele obiectelor) peste punctele radar, pe grila BEV 192x192.
Rulare:  python scripts/02_dataset_targets.py --split val --index 0
"""
import argparse

from _env import setup, OUT_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["train", "val"])
ap.add_argument("--index", type=int, default=0)
args = ap.parse_args()

setup()  # inainte de importurile CARMA!

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from config import cfg
from processing import NuScenesDataset

ds = NuScenesDataset(args.split)
print(f"\nDataset '{args.split}': {len(ds)} sample-uri")
item = ds[args.index]

print("\n=== Ce contine un sample ===")
for k, v in item.items():
    if torch.is_tensor(v):
        print(f"  {k:28s} {str(tuple(v.shape)):22s} {str(v.dtype):14s} "
              f"min={v.float().min().item():8.3f} max={v.float().max().item():8.3f}")
    else:
        print(f"  {k:28s} {type(v).__name__}")

pr = cfg.data.point_cloud_range
H, W = cfg.data.bev_h, cfg.data.bev_w
cell = (pr[3] - pr[0]) / H
names = cfg.head.class_names
print(f"\nGrila BEV: {H}x{W}, celula = {cell:.3f} m, acoperire x,y in [{pr[0]}, {pr[3]}] m")


def to_grid(x, y):
    """metri (ego) -> (coloana, rand) in grila. Convenția CARMA: rand = x (inainte), coloana = y."""
    return (y - pr[1]) / cell, (x - pr[0]) / cell


radar = item["radar"]
valid = radar.abs().sum(1) > 0
rad = radar[valid]
print(f"Radar: {int(valid.sum())} puncte valide din {radar.shape[0]} (restul = padding cu zero)")
print("  coloanele 0,1 = x,y in metri; 3 = RCS; 6,7,8 = distanta/viteza/recenta normalizate")

heat = item["heatmaps"]                    # (10, H, W)
gt_c = item["gt_center"]
gt_k = item["gt_class_idx"]
print(f"GT: {len(gt_k)} obiecte -> " + ", ".join(
    f"{names[c]}={int((gt_k == c).sum())}" for c in range(len(names)) if (gt_k == c).any()))

# --- Figura 1: heatmap tinta (max pe clase) + radar + centre GT ---
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(heat.max(0).values.numpy(), origin="lower", cmap="magma")
rc, rr = to_grid(rad[:, 0].numpy(), rad[:, 1].numpy())
ax.scatter(rc, rr, s=4, c="cyan", label="radar")
if len(gt_c):
    gc, gr = to_grid(gt_c[:, 0].numpy(), gt_c[:, 1].numpy())
    ax.scatter(gc, gr, s=30, facecolors="none", edgecolors="lime", label="centre GT")
ax.scatter([W / 2], [H / 2], marker="^", s=120, c="white", label="ego")
ax.invert_xaxis()  # ca +y (stanga masinii) sa apara in stanga; inainte = sus
ax.set_title("Heatmap tinta (max peste clase) + radar")
ax.legend(loc="upper right")
p1 = OUT_DIR / f"02_{args.split}{args.index:04d}_heatmap_radar.png"
fig.savefig(p1, dpi=120, bbox_inches="tight")

# --- Figura 2: cate un heatmap pe clasa ---
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for c, a in enumerate(axes.flat):
    a.imshow(heat[c].numpy(), origin="lower", cmap="magma", vmin=0, vmax=1)
    a.invert_xaxis()
    a.set_title(f"{names[c]}  (n={int((gt_k == c).sum())})")
    a.axis("off")
p2 = OUT_DIR / f"02_{args.split}{args.index:04d}_heatmap_per_class.png"
fig.savefig(p2, dpi=100, bbox_inches="tight")

# --- regresia: ce valori se invata in centrul unui obiect ---
mask = item["reg_masks"]
reg = item["reg_targets"]
print(f"\nreg_targets {tuple(reg.shape)} | reg_masks {tuple(mask.shape)}")
print("  cele 10 canale: dx, dy, z, w, l, h, sin(yaw), cos(yaw), vx, vy")
m = mask[0] if mask.dim() == 3 else mask
idx = (m > 0).nonzero()
if len(idx):
    r, c = idx[0].tolist()
    print(f"  exemplu in celula (rand={r}, col={c}): {[round(v, 3) for v in reg[:, r, c].tolist()]}")

print(f"\nSalvat:\n  {p1}\n  {p2}")
