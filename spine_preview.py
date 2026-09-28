"""spine_preview — render a keyframe-montage PNG of a rig's animations WITHOUT a
browser or game. Composites atlas parts at chosen anim times applying bone
translate/rotate/scale + attachment swaps. Bone transforms are inherited down
the hierarchy as affine matrices, so parts pivot about their bones like the
runtime (limb chains need that). Keys are interpolated linearly, so curves are
approximated — a fast check of assembly, draw order and poses, not a final render.
"""
from __future__ import annotations
import json, math, os
from PIL import Image, ImageDraw


def _affine(x, y, rot, sx, sy):
    """3x3 matrix for translate(x, y) · rotate(rot degrees) · scale(sx, sy)."""
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    return [[c * sx, -s * sy, x], [s * sx, c * sy, y], [0.0, 0.0, 1.0]]


def _mul(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]


def _inv(m):
    a, b, tx = m[0]; c, d, ty = m[1]
    det = a * d - b * c
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    return [[ia, ib, -(ia * tx + ib * ty)], [ic, id_, -(ic * tx + id_ * ty)], [0.0, 0.0, 1.0]]


def _bone_worlds(bones_list, bd):
    """World matrix per bone: setup transform + animation offsets, parent first."""
    W = {}
    for b in bones_list:
        e = bd.get(b["name"], {"rot": 0, "tx": 0, "ty": 0, "sx": 1, "sy": 1})
        local = _affine(b.get("x", 0) + e["tx"], b.get("y", 0) + e["ty"], b.get("rotation", 0) + e["rot"],
                        b.get("scaleX", 1) * e["sx"], b.get("scaleY", 1) * e["sy"])
        p = b.get("parent")
        W[b["name"]] = _mul(W[p], local) if p else local
    return W


def _lerp(kf, t):
    if t <= kf[0]["time"]:
        return kf[0]
    if t >= kf[-1]["time"]:
        return kf[-1]
    for i in range(len(kf) - 1):
        a, b = kf[i], kf[i + 1]
        if a["time"] <= t <= b["time"]:
            f = (t - a["time"]) / (b["time"] - a["time"])
            return {k: (a.get(k, 0) + (b.get(k, 0) - a.get(k, 0)) * f
                        if isinstance(a.get(k, 0), (int, float)) and isinstance(b.get(k, 0), (int, float))
                        else a.get(k))
                    for k in (set(a) | set(b)) - {"curve"}}
    return kf[-1]


def _render(d, images_dir, anim, t, maxpx):
    sk = d["skeleton"]
    slots = d["slots"]; att = d["skins"][0]["attachments"]; A = d["animations"][anim]
    W, H = int(sk["width"]), int(sk["height"])
    pad = int(0.2 * max(W, H))   # room for limbs swung past the setup bounds
    CW, CH = W + 2 * pad, H + 2 * pad
    cv = Image.new("RGBA", (CW, CH), (24, 20, 32, 255))
    cur = {s["name"]: s.get("attachment") for s in slots}
    for sn, ad in A.get("slots", {}).items():
        if "attachment" in ad:
            name = ad["attachment"][0]["name"]
            for k in ad["attachment"]:
                if k["time"] <= t:
                    name = k["name"]
            cur[sn] = name
    bd = {}
    for bn, tl in A.get("bones", {}).items():
        e = {"rot": 0, "tx": 0, "ty": 0, "sx": 1, "sy": 1}
        if "rotate" in tl:
            e["rot"] = _lerp(tl["rotate"], t).get("value", 0)
        if "translate" in tl:
            v = _lerp(tl["translate"], t); e["tx"], e["ty"] = v.get("x", 0), v.get("y", 0)
        if "scale" in tl:
            v = _lerp(tl["scale"], t); e["sx"], e["sy"] = v.get("x", 1), v.get("y", 1)
        bd[bn] = e
    worlds = _bone_worlds(d["bones"], bd)
    for s in slots:
        region = cur[s["name"]]
        if not region:
            continue
        ent = att[s["name"]][region]; path = ent.get("path", region)
        p = os.path.join(images_dir, f"{path}.png")
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGBA")
        # attachment w/h give the DISPLAY size — the png itself may be larger
        if (im.width, im.height) != (ent.get("width", im.width), ent.get("height", im.height)):
            im = im.resize((max(1, int(ent.get("width", im.width))),
                            max(1, int(ent.get("height", im.height)))), Image.LANCZOS)
        # image pixel (u, v) → attachment local → bone world → canvas pixel
        to_att = [[1.0, 0.0, -im.width / 2], [0.0, -1.0, im.height / 2], [0.0, 0.0, 1.0]]
        att_m = _affine(ent.get("x", 0), ent.get("y", 0), ent.get("rotation", 0), 1, 1)
        to_canvas = [[1.0, 0.0, W / 2 + pad], [0.0, -1.0, H + pad], [0.0, 0.0, 1.0]]
        m = _mul(to_canvas, _mul(worlds[s["bone"]], _mul(att_m, to_att)))
        inv = _inv(m)
        warped = im.transform((CW, CH), Image.AFFINE, (*inv[0], *inv[1]), resample=Image.BICUBIC)
        cv.alpha_composite(warped)
    sc = maxpx / max(CW, CH)
    return cv.resize((max(1, int(CW * sc)), max(1, int(CH * sc))))


# default poses to sample per animation (time in seconds)
_POSES = {"idle": [0.0, 1.4], "win": [0.1, 0.28, 0.5], "blink": [0.15], "pop": [0.0]}


def montage(rig_json: str, images_dir: str, out_png: str, maxpx: int = 200) -> str:
    d = json.load(open(rig_json))
    shots = []
    for a in d["animations"]:
        for t in _POSES.get(a, [0.0]):
            shots.append((a, t))
    cells = [(_render(d, images_dir, a, t, maxpx), f"{a}@{t}") for a, t in shots]
    cw, ch = cells[0][0].size
    cols = len(cells)
    mont = Image.new("RGBA", (cols * cw + 8 * (cols + 1), ch + 22), (0, 0, 0, 255))
    dr = ImageDraw.Draw(mont)
    for i, (im, lbl) in enumerate(cells):
        x = 8 + i * (cw + 8)
        mont.paste(im, (x + (cw - im.width) // 2, 10 + (ch - im.height) // 2))
        dr.text((x + 2, ch + 8), lbl, fill=(255, 255, 255, 255))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    mont.convert("RGB").save(out_png)
    return out_png
