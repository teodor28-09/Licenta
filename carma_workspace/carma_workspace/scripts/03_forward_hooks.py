"""
Pasul 3 - modelul antrenat, etapa cu etapa.

Ce vezi: shape-ul si "harta de activare" dupa fiecare bloc
(camera BEV, radar BEV, CAMF, neck), heatmap-ul prezis si detectiile.
Cu --zero-radar / --zero-camera faci ablatia pe senzori (teza, 4.11).

Rulare:
  python scripts/03_forward_hooks.py --index 0
  python scripts/03_forward_hooks.py --index 0 --zero-radar
  python scripts/03_forward_hooks.py --index 0 --zero-camera
"""
import argparse

from _env import setup, OUT_DIR, CHECKPOINT

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["train", "val"])
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
ap.add_argument("--score", type=float, default=0.40, help="prag de scor (teza: 0.40)")
ap.add_argument("--zero-radar", action="store_true")
ap.add_argument("--zero-camera", action="store_true")
args = ap.parse_args()

setup()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from config import cfg
from processing import NuScenesDataset
from model_fusion import CARMADet
from utility import _decode_for_validation, _post_center_nms

dev = torch.device("cuda" if (args.device == "cuda" or
                              (args.device == "auto" and torch.cuda.is_available())) else "cpu")
print(f"[device] {dev}")

# ---------- model + checkpoint (incarcare tolerantă, ca in scripturile din repo) ----------
model = CARMADet().to(dev).eval()
ckpt = torch.load(CHECKPOINT, map_location=dev, weights_only=False)
state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
ms = model.state_dict()
ok = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
ms.update(ok)
model.load_state_dict(ms)
print(f"[ckpt] {len(ok)}/{len(ms)} tensori incarcati | epoca={ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'}")
n_params = sum(p.numel() for p in model.parameters())
print(f"[model] {n_params / 1e6:.1f} M parametri")
for name, mod in model.named_children():
    print(f"   {name:32s} {sum(p.numel() for p in mod.parameters()) / 1e6:7.2f} M")

# ---------- hooks: retinem iesirea fiecarui bloc important ----------
captured = {}


def first_4d(x):
    if torch.is_tensor(x) and x.dim() == 4:
        return x
    if isinstance(x, (tuple, list)):
        for e in x:
            t = first_4d(e)
            if t is not None:
                return t
    if isinstance(x, dict):
        for e in x.values():
            t = first_4d(e)
            if t is not None:
                return t
    return None


def make_hook(name):
    def hook(_m, _inp, out):
        t = first_4d(out)
        if t is not None:
            captured[name] = t.detach().float().cpu()
    return hook


for name in ["camera_encoder", "camera_bev_transformer", "radar_encoder", "camf", "bev_neck", "bev_transformer"]:
    if hasattr(model, name):
        getattr(model, name).register_forward_hook(make_hook(name))

# ---------- un sample ----------
ds = NuScenesDataset(args.split)
item = ds[args.index]
imgs = item["imgs"].unsqueeze(0).to(dev)
cam2ego = item["cam2ego"].unsqueeze(0).to(dev)
intr = item["intrinsics"].unsqueeze(0).to(dev)
radar = item["radar"].unsqueeze(0).to(dev)

tag = "normal"
if args.zero_radar:
    radar = torch.zeros_like(radar); tag = "fara_radar"
if args.zero_camera:
    imgs = torch.zeros_like(imgs); tag = "fara_camera"  # 0 dupa normalizare = imagine "gri medie"

with torch.no_grad():
    out = model(imgs, cam2ego, intr, radar, return_aux=True)
pred_heat, pred_reg = out[0], out[1]

print(f"\n=== Ce iese din fiecare bloc ({tag}) ===")
for k, t in captured.items():
    print(f"  {k:26s} {str(tuple(t.shape)):22s} medie|x|={t.abs().mean():.4f}")
print(f"  {'pred_heat (logits)':26s} {str(tuple(pred_heat.shape))}")
print(f"  {'pred_reg':26s} {str(tuple(pred_reg.shape))}")

# ---------- decodare: heatmap -> cutii ----------
dets = _decode_for_validation(pred_heat[0], pred_reg[0], score_thr=args.score)
dets = _post_center_nms(dets)
names = cfg.head.class_names
gt_k = item["gt_class_idx"]
print(f"\n=== Detectii la prag {args.score} ===  total={len(dets)} | GT={len(gt_k)}")
for c, n in enumerate(names):
    nd = sum(d["class_idx"] == c for d in dets)
    ng = int((gt_k == c).sum())
    if nd or ng:
        print(f"  {n:22s} prezise={nd:3d}  GT={ng:3d}")
for d in dets[:10]:
    print(f"   {d['class']:20s} scor={d['score']:.2f} centru={[round(v, 1) for v in d['center']]}")

# ---------- figura: ce "vede" fiecare etapa ----------
panels = [(k, captured[k][0].abs().mean(0).numpy()) for k in captured]
panels.append(("heatmap prezis (max clase)", pred_heat[0].float().sigmoid().max(0).values.cpu().numpy()))
panels.append(("heatmap tinta (GT)", item["heatmaps"].max(0).values.numpy()))
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.5))
for a, (title, img) in zip(axes, panels):
    a.imshow(img, origin="lower", cmap="viridis")
    a.invert_xaxis()
    a.set_title(f"{title}\n{img.shape[0]}x{img.shape[1]}", fontsize=9)
    a.axis("off")
fig.suptitle(f"CARMA - sample {args.split}[{args.index}] - {tag}")
p = OUT_DIR / f"03_{args.split}{args.index:04d}_{tag}.png"
fig.savefig(p, dpi=110, bbox_inches="tight")
print(f"\nSalvat: {p}")
