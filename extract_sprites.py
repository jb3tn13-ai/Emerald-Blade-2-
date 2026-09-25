"""Segment white-background sprite sheets into individual sprites with alpha."""
import os
import json
import numpy as np
from PIL import Image
from scipy import ndimage

SHEETS = ["knight.png", "bad guys.png", "villian.png", "castle.png", "items.png"]
OUT = "sprites"
os.makedirs(OUT, exist_ok=True)

def segment(path, name):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    # background = near the corner color (handles off-white tints)
    corner = a[2, 2].astype(np.int16)
    dist = np.sqrt(((a - corner.astype(np.int32)) ** 2).sum(axis=2))
    bg = dist < 32
    mask = ~bg
    # close small gaps (connect weapon to hand etc.)
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
    lab, n = ndimage.label(mask)
    objs = ndimage.find_objects(lab)
    results = []
    idx = 0
    for sl in objs:
        if sl is None:
            continue
        ys, xs = sl
        h = ys.stop - ys.start
        w = xs.stop - xs.start
        if w * h < 200:  # skip specks / caption text fragments
            continue
        sub = a[ys.start:ys.stop, xs.start:xs.stop].astype(np.uint8)
        sub_dist = np.sqrt(((sub.astype(np.int32) - corner) ** 2).sum(axis=2))
        sub_bg = sub_dist < 32
        alpha = np.where(sub_bg, 0, 255).astype(np.uint8)
        # keep interior whites opaque: flood fill background from borders only
        border = np.zeros_like(sub_bg)
        border[0, :] = sub_bg[0, :]
        border[-1, :] = sub_bg[-1, :]
        border[:, 0] = sub_bg[:, 0]
        border[:, -1] = sub_bg[:, -1]
        ext, _ = ndimage.label(sub_bg)
        border_labels = set(np.unique(ext[border])) - {0}
        conn = np.isin(ext, list(border_labels))
        alpha = np.where(conn, 0, 255).astype(np.uint8)
        img = np.dstack([sub, alpha])
        out = Image.fromarray(img, "RGBA")
        fname = f"{name}_{idx:03d}.png"
        out.save(os.path.join(OUT, fname))
        results.append({"file": fname, "x": int(xs.start), "y": int(ys.start),
                        "w": int(w), "h": int(h), "area": int(w * h)})
        idx += 1
    return results

manifest = {}
for s in SHEETS:
    name = s.replace(".png", "").replace(" ", "_")
    manifest[name] = segment(s, name)
    print(s, "->", len(manifest[name]), "sprites")

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("done")
