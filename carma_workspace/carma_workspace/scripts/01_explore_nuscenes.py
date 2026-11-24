"""
Pasul 1 - datele brute nuScenes (fara CARMA).

Ce vezi: 6 camere, radarul, adnotarile 3D ale unui sample.
Rulare:  python scripts/01_explore_nuscenes.py --index 0
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # salveaza in fisiere, fara ferestre
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

from _env import DATA_ROOT, VERSION, OUT_DIR

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
          "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, default=0, help="indexul sample-ului din nusc.sample")
args = ap.parse_args()

nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
out = OUT_DIR / f"01_sample{args.index:04d}"
out.mkdir(parents=True, exist_ok=True)

print("\n=== Scene ===")
for s in nusc.scene:
    print(f"  {s['name']:12s} {s['nbr_samples']:3d} keyframes | {s['description']}")

sample = nusc.sample[args.index]
print(f"\n=== Sample {args.index} | token={sample['token']} ===")

# 1) camerele: imagine + calibrare
for cam in CAMS:
    sd = nusc.get("sample_data", sample["data"][cam])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.array(cs["camera_intrinsic"])
    print(f"  {cam:16s} {sd['width']}x{sd['height']}  fx={K[0,0]:.0f}  "
          f"pozitie fata de ego={np.round(cs['translation'], 2)}")
    nusc.render_sample_data(sd["token"], out_path=str(out / f"{cam}.png"), verbose=False)

# 2) radarul: cate puncte si ce campuri are fiecare punct
print("\n  Radar (18 campuri brute/punct: x y z dyn_prop id rcs vx vy vx_comp vy_comp ...)")
total = 0
for rad in RADARS:
    sd = nusc.get("sample_data", sample["data"][rad])
    pc = RadarPointCloud.from_file(f"{DATA_ROOT}/{sd['filename']}")
    n = pc.points.shape[1]
    total += n
    rcs = pc.points[5]
    print(f"  {rad:18s} {n:4d} puncte | RCS min/max = {rcs.min() if n else 0:.1f}/{rcs.max() if n else 0:.1f}")
print(f"  TOTAL radar: {total} puncte (CARMA pastreaza max 3000 si le codifica in 13 feature-uri)")
nusc.render_sample_data(sample["data"]["RADAR_FRONT"], nsweeps=1, underlay_map=False,
                        out_path=str(out / "RADAR_FRONT_bev.png"), verbose=False)

# 3) adnotarile = ce trebuie sa prezica modelul
print(f"\n  Adnotari: {len(sample['anns'])}")
for tok in sample["anns"][:15]:
    a = nusc.get("sample_annotation", tok)
    print(f"   {a['category_name']:38s} centru(global)={np.round(a['translation'], 1)} "
          f"marime(w,l,h)={np.round(a['size'], 1)}")
if len(sample["anns"]) > 15:
    print("   ...")

# 4) toate senzorii impreuna
nusc.render_sample(sample["token"], out_path=str(out / "all_sensors.png"), verbose=False)
print(f"\nImagini salvate in: {out}")
