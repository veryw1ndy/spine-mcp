"""spine_rig — turn a cut-up character into a rigged + animated Spine 4.2 skeleton.

Input is EITHER:
  - a PhotoshopToSpine export folder  (<dir>/<name>.json + <dir>/images/*.png), or
  - a layered .psd                    (each top-level layer = one part; read via psd-tools)

Output (written to out_dir):
  <name>.json   runtime skeleton (bones/slots/skin/animations), Spine 4.2 format
  <name>.atlas  texture atlas (shelf-packed; region name == attachment name)
  <name>.png    atlas page

Rig: body(root) + head(neck) [+ rot]; collar/fire ride the body. Head-state
families (head/head_win/head_blink or face/face_win/face_blink) collapse into ONE
slot with attachment-swap inside the win/blink timelines.

Limb layers (see LIMB_WORDS) get a bone chain per side: upper_arm → forearm →
hand on the body, thigh → shin → foot on a `hips` bone. Joints come from each
part's opaque pixels, so angled limbs pivot correctly. With legs present, hips
becomes the parent of body and legs and carries every body translate, so a jump
lifts the legs too; rotate and squash stay on the upper body. Without limb
layers the output is exactly the body + head rig above.

Anims: idle(loop) · win(squash-stretch pop + face-swap) · blink(face-swap) ·
pop(squash landing). One-shots end at the setup pose so they mix back cleanly.
Arms sway in idle and swing outward in win.

This is the engine behind the Spine MCP server's `rig_and_animate` tool; it is a
pure function (no MCP, no globals) so it can also be imported or run standalone.
"""
from __future__ import annotations
import json, math, os, glob, re
import numpy as np
from PIL import Image

SUFFIX = ("_win", "_blink")


# ---------------------------------------------------------------- input readers
def _read_photoshop_export(export_dir: str):
    """Return (parts, draw_order, images_dir). parts[slot]=dict(cx,cy,w,h,file)."""
    exp = None
    for cand in sorted(glob.glob(f"{export_dir}/*.json")):
        try:
            data = json.load(open(cand))
        except Exception:
            continue
        if isinstance(data, dict) and "skins" in data:   # the layout, not bulges.json etc.
            exp = data
            break
    if exp is None:
        raise ValueError(f"no layout json with 'skins' in {export_dir}")
    images_dir = f"{export_dir}/images"
    skin = exp["skins"]["default"] if isinstance(exp["skins"], dict) else \
        next(s for s in exp["skins"] if s["name"] == "default")["attachments"]
    draw = [s["name"] for s in exp["slots"]]                       # back→front
    parts = {}
    for slot, atts in skin.items():
        an, pl = next(iter(atts.items()))
        parts[slot] = dict(x=float(pl.get("x", 0)), y=float(pl.get("y", 0)),
                            w=float(pl["width"]), h=float(pl["height"]), file=an)
    return parts, draw, images_dir


def _read_psd(psd_path: str, work_dir: str):
    """Flatten each top-level PSD layer to a PNG and record its centre/size.
    Mirrors what PhotoshopToSpine does, so the rest of the pipeline is identical."""
    from psd_tools import PSDImage
    psd = PSDImage.open(psd_path)
    images_dir = os.path.join(work_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    Wc, Hc = psd.width, psd.height
    parts, draw = {}, []
    for layer in psd:                                             # bottom→top order
        if not layer.is_visible() or layer.bbox == (0, 0, 0, 0):
            continue
        name = layer.name.strip()
        img = layer.composite()
        if img is None:
            continue
        img.save(os.path.join(images_dir, f"{name}.png"))
        l, t, r, b = layer.bbox
        w, h = r - l, b - t
        cx = (l + r) / 2 - Wc / 2                                 # centre, origin mid-top
        cy = Hc - (t + b) / 2                                     # +Y up from bottom
        parts[name] = dict(x=cx, y=cy, w=float(w), h=float(h), file=name)
        draw.append(name)
    return parts, draw, images_dir


# ------------------------------------------------------------------- rig engine
# Which part names count as the head, beyond the ones detected as a state family.
#
# A heuristic, and one tuned on one studio's art — "crown" and "tooth" are head
# parts for a slot symbol and would be nothing of the kind on a knight. A layer
# called kopf, cabeza or tete lands in the body with no complaint, which is the
# failure worth knowing about rather than pretending away.
#
# Overridable so a different naming convention does not need a fork:
#   SPINE_HEAD_WORDS=kopf,gesicht,krone
HEAD_WORDS = tuple(
    w.strip().lower()
    for w in os.environ.get("SPINE_HEAD_WORDS", "head,face,golova,crown,tooth").split(",")
    if w.strip()
)


# Which part names count as limb segments. There is no team naming spec for limb
# layers yet, so this table is the contract. English names are split into words
# ("upper_arm_l", "LeftForearm", "hand-R") and matched word by word, so "armor" or
# "charm" never count as an arm. Chinese names are matched as substrings, longest
# first. A side word is optional; without one the side comes from which half of
# the torso the part sits on ("l" = viewer's left, same as the layer names).
LIMB_WORDS = {
    "upperarm": ("arm", 0), "arm": ("arm", 0),
    "forearm": ("arm", 1), "lowerarm": ("arm", 1),
    "hand": ("arm", 2),
    "thigh": ("leg", 0), "upperleg": ("leg", 0), "leg": ("leg", 0),
    "shin": ("leg", 1), "calf": ("leg", 1), "lowerleg": ("leg", 1),
    "foot": ("leg", 2),
}
LIMB_ZH = (("手臂", "arm", 0), ("胳膊", "arm", 0), ("大臂", "arm", 0), ("上臂", "arm", 0),
           ("小臂", "arm", 1), ("前臂", "arm", 1), ("大腿", "leg", 0), ("小腿", "leg", 1),
           ("手", "arm", 2), ("臂", "arm", 0), ("腿", "leg", 0), ("脚", "leg", 2), ("足", "leg", 2))
# Both limbs painted on one layer cannot get a bone each; it stays on the body.
MERGED_LIMB_WORDS = ("arms", "legs")
MERGED_LIMB_ZH = ("双臂", "双腿", "两腿", "双手")
SIDE_WORDS = {"l": "l", "left": "l", "r": "r", "right": "r", "front": "front", "back": "back"}
SEG_NAMES = {"arm": ("upper_arm", "forearm", "hand"), "leg": ("thigh", "shin", "foot")}
TORSO_WORDS = ("body", "torso", "chest", "身体", "躯干")


def _words(name: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    w = [t for t in re.split(r"[^a-z]+", s.lower()) if t]
    out, i = [], 0
    while i < len(w):   # glue "upper arm" / "lower leg" into one word
        if i + 1 < len(w) and w[i] in ("upper", "lower") and w[i + 1] in ("arm", "leg"):
            out.append(w[i] + w[i + 1]); i += 2
        else:
            out.append(w[i]); i += 1
    return out


def _limb_of(name: str):
    """{"limb", "seg", "side"} for a limb layer, "merged" for a both-limbs layer,
    None otherwise. side may be None (resolved later from position)."""
    words = _words(name)
    if any(w in MERGED_LIMB_WORDS for w in words) or any(k in name for k in MERGED_LIMB_ZH):
        return "merged"
    hit = next((LIMB_WORDS[w] for w in words if w in LIMB_WORDS), None)
    if hit is None:
        hit = next(((limb, seg) for k, limb, seg in LIMB_ZH if k in name), None)
    if hit is None:
        return None
    side = next((SIDE_WORDS[w] for w in words if w in SIDE_WORDS), None)
    if side is None:
        side = "l" if "左" in name else "r" if "右" in name else None
    return {"limb": hit[0], "seg": hit[1], "side": side}


def _classify(name: str, state_bases) -> str:
    s = name.lower()
    if s in state_bases:
        return "head"
    if any(k in s for k in HEAD_WORDS):
        return "head"
    if isinstance(_limb_of(name), dict):
        return "limb"
    if "rot" in s:
        return "rot"
    return "body"


def _limb_axis(img: Image.Image, part: dict) -> dict | None:
    """Medial axis of a limb part from its opaque pixels, in rig coords (+Y up).
    Each end is the outermost point along the long axis, placed across the axis
    at the mean of the last 8 % of pixels, so it sits on the limb's centre line
    even when the limb is drawn at an angle."""
    alpha = np.asarray(img.getchannel("A")) > 32
    ys, xs = np.nonzero(alpha)
    if len(xs) < 8:
        return None
    sx, sy = part["w"] / img.width, part["h"] / img.height
    X = part["cx"] - part["w"] / 2 + (xs + 0.5) * sx
    Y = part["cy"] + part["h"] / 2 - (ys + 0.5) * sy
    P = np.stack([X, Y], 1)
    m = P.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((P - m).T))
    d = evecs[:, int(np.argmax(evals))]
    t = (P - m) @ d
    nrm = (P - m) @ np.array([-d[1], d[0]])
    lo, hi = float(t.min()), float(t.max())
    band = 0.08 * (hi - lo)
    perp = np.array([-d[1], d[0]])
    e0 = m + d * lo + perp * float(nrm[t <= lo + band].mean())
    e1 = m + d * hi + perp * float(nrm[t >= hi - band].mean())
    width = float(np.percentile(nrm, 95) - np.percentile(nrm, 5))
    return {"ends": (e0, e1), "width": width, "length": hi - lo}


def build_rig(source: str, out_dir: str, name: str | None = None,
              kind: str = "symbol", anims: list[str] | None = None) -> dict:
    """Build the skeleton. `source` is an export folder or a .psd. Returns a
    summary dict {name, width, height, bones, slots, head_slot, variants, anims,
    files}."""
    if name is None:
        name = os.path.splitext(os.path.basename(source.rstrip("/")))[0]
    os.makedirs(out_dir, exist_ok=True)

    if source.lower().endswith(".psd"):
        parts, draw, images_dir = _read_psd(source, out_dir)
    else:
        parts, draw, images_dir = _read_photoshop_export(source)
    if not parts:
        raise ValueError(f"no parts found in {source}")

    # detect head-state families: base + base_win + base_blink
    fam = {}
    for n in draw:
        base = n
        for s in SUFFIX:
            if n.lower().endswith(s):
                base = n[: -len(s)]
                break
        state = next((s[1:] for s in SUFFIX if n.lower().endswith(s)), "base")
        fam.setdefault(base, {})[state] = n
    STATE_FAM = {b: v for b, v in fam.items() if len(v) > 1 and "base" in v}
    variant2base = {sl: b for b, st in STATE_FAM.items() for k, sl in st.items() if k != "base"}
    draw_final = [n for n in draw if n not in variant2base]
    state_bases = {b.lower() for b in STATE_FAM}

    # normalize: root at bottom-centre, +Y up
    minX = min(p["x"] - p["w"] / 2 for p in parts.values())
    maxX = max(p["x"] + p["w"] / 2 for p in parts.values())
    minY = min(p["y"] - p["h"] / 2 for p in parts.values())
    maxY = max(p["y"] + p["h"] / 2 for p in parts.values())
    cx0 = (minX + maxX) / 2
    W, H = maxX - minX, maxY - minY
    norm = {n: dict(cx=p["x"] - cx0, cy=p["y"] - minY, w=p["w"], h=p["h"], file=p["file"])
            for n, p in parts.items()}

    cls = {n: _classify(n, state_bases) for n in draw_final}
    FIRE = {n for n in draw_final if "fire" in n.lower()}
    # light-emitting FX layers render additive but do NOT behave like fire
    # (no ignite/idle heat pulse — the animator lights them in win/destroy)
    _ADD = ("fx_flash", "fx_spark", "fx_bolt", "fx_ring", "fx_star", "fx_glow")
    ADDITIVE_FX = {n for n in draw_final if any(k in n.lower() for k in _ADD)}
    has_head = any(v == "head" for v in cls.values())
    has_rot = any(v == "rot" for v in cls.values())

    heads = [n for n in draw_final if cls[n] == "head"]
    hx = sum(norm[n]["cx"] for n in heads) / len(heads) if heads else 0
    hy = (sum(norm[n]["cy"] for n in heads) / len(heads)
          - max(norm[n]["h"] for n in heads) * 0.42) if heads else H * 0.6
    rots = [n for n in draw_final if cls[n] == "rot"]
    all_regions = list(norm.keys())
    imgs = {n: Image.open(f"{images_dir}/{norm[n]['file']}.png").convert("RGBA") for n in all_regions}

    # ---- limb chains ---------------------------------------------------------
    warnings = []
    for n in draw_final:
        if cls[n] == "body" and _limb_of(n) == "merged":
            warnings.append(f"{n}: both limbs on one layer, left on the body; "
                            "split it into one layer per side to get limb bones")
    body_parts = [n for n in draw_final if cls[n] == "body"]
    torso = ([n for n in body_parts if any(k in n.lower() for k in TORSO_WORDS)]
             or [n for n in body_parts if _limb_of(n) is None] or body_parts or draw_final)
    tx0 = min(norm[n]["cx"] - norm[n]["w"] / 2 for n in torso)
    tx1 = max(norm[n]["cx"] + norm[n]["w"] / 2 for n in torso)
    ty0 = min(norm[n]["cy"] - norm[n]["h"] / 2 for n in torso)
    ty1 = max(norm[n]["cy"] + norm[n]["h"] / 2 for n in torso)
    tcx = (tx0 + tx1) / 2
    # A chain's root end is the one nearer the shoulder line (arms) or the hip
    # line (legs), which holds for hanging, raised and sideways limbs alike.
    ANCHOR = {"arm": np.array([tcx, ty1]), "leg": np.array([tcx, ty0])}

    chains = {}   # (limb, side) -> {seg: [part names]}
    for n in draw_final:
        if cls[n] != "limb":
            continue
        info = _limb_of(n)
        side = info["side"] or ("l" if norm[n]["cx"] < tcx else "r")
        chains.setdefault((info["limb"], side), {}).setdefault(info["seg"], []).append(n)

    LIMB_BONE, LIMB_CHAINS, OUTWARD, limb_specs = {}, {}, {}, []
    for (limb, side), segs in sorted(chains.items()):
        joints = []   # (seg, members, joint, far end)
        prev_far = None
        for seg in sorted(segs):
            members = segs[seg]
            geo = max(members, key=lambda k: int((np.asarray(imgs[k].getchannel("A")) > 32).sum()))
            ax = _limb_axis(imgs[geo], norm[geo])
            if ax is None:
                for k in members:
                    cls[k] = "body"
                warnings.append(f"{geo}: too few opaque pixels for a limb axis, left on the body")
                continue
            e0, e1 = ax["ends"]
            ref = ANCHOR[limb] if prev_far is None else prev_far
            near, far = (e0, e1) if np.linalg.norm(e0 - ref) <= np.linalg.norm(e1 - ref) else (e1, e0)
            # Joints sit at the centre of the rounded end cap, half a limb width
            # in from the tip; between segments, midway between the two caps,
            # so an elbow or ankle bent at 90 degrees still lands on the joint.
            inward = (far - near) / max(float(np.linalg.norm(far - near)), 1e-6)
            cap = min(ax["width"] / 2, 0.3 * ax["length"])
            near_cap, far_cap = near + inward * cap, far - inward * cap
            joint = near_cap if prev_far is None else (prev_far_cap + near_cap) / 2
            joints.append((seg, members, joint, far))
            prev_far, prev_far_cap = far, far_cap
        if not joints:
            continue
        single = len(joints) == 1 and joints[0][0] == 0
        parent = "body" if limb == "arm" else "hips"
        names = []
        for i, (seg, members, joint, far) in enumerate(joints):
            tip = joints[i + 1][2] if i + 1 < len(joints) else far
            d = tip - joint
            bn = f"{limb}_{side}" if single else f"{SEG_NAMES[limb][seg]}_{side}"
            limb_specs.append((bn, parent, joint, math.degrees(math.atan2(d[1], d[0])),
                               float(np.linalg.norm(d))))
            for k in members:
                LIMB_BONE[k] = bn
            if i == 0:
                u = d / max(float(np.linalg.norm(d)), 1e-6)
                # +rotation moves the tip along (-u.y, u.x): outward when that
                # points away from the centre line; sideways limbs raise instead
                s = -u[1] * (tip[0] - tcx) if abs(u[1]) > 0.5 else u[0]
                OUTWARD[bn] = 1 if s >= 0 else -1
            names.append(bn)
            parent = bn
        LIMB_CHAINS[f"{limb}_{side}"] = names
    has_legs = any(p == "hips" for _b, p, *_ in limb_specs)

    # BONES[name] = world transform (x, y, rotation in degrees) + parent + length
    def _bone(parent, x, y, rot=0.0, length=0.0):
        return {"parent": parent, "x": round(float(x), 2), "y": round(float(y), 2),
                "rot": round(float(rot), 2), "length": round(float(length), 2)}

    if has_legs:
        hip_y = sum(float(j[1]) for _b, p, j, *_ in limb_specs if p == "hips") / \
            sum(1 for _b, p, *_ in limb_specs if p == "hips")
        BONES = {"hips": _bone("root", tcx, hip_y), "body": _bone("hips", tcx, hip_y)}
    else:
        BONES = {"body": _bone("root", 0.0, H * 0.30)}
    if has_head:
        BONES["head"] = _bone("body", hx, hy)
    if has_rot:
        BONES["rot"] = _bone("root", norm[rots[0]]["cx"], norm[rots[0]]["cy"])
    for bn, parent, joint, rot, length in limb_specs:
        BONES[bn] = _bone(parent, joint[0], joint[1], rot, length)

    def slotbone(n):
        if cls[n] == "limb":
            return LIMB_BONE[n]
        if has_rot and cls[n] == "rot":
            return "rot"
        if has_head and cls[n] == "head":
            return "head"
        return "body"

    def bworld(n):
        return (0.0, 0.0, 0.0) if n == "root" else (BONES[n]["x"], BONES[n]["y"], BONES[n]["rot"])

    def to_local(x, y, frame):
        """World point → local coords of a bone frame (x, y, rotation)."""
        fx, fy, fr = frame
        c, s = math.cos(math.radians(-fr)), math.sin(math.radians(-fr))
        dx, dy = x - fx, y - fy
        return dx * c - dy * s, dx * s + dy * c

    # ---- atlas (shelf-pack; region name == attachment name) ------------------
    MAXW, PAD = 1024, 2
    # A part WIDER than the page used to be pasted anyway: PIL crops silently at
    # the page edge while the .atlas still declares the full region size, so the
    # overhanging columns sample outside the texture and the renderer clamps them
    # into a smeared strip (Soul Siphon: a 1070px frame on a 1024px page → the
    # right 48px of the bezel was a stretched streak). Grow the page instead.
    MAXW = max(MAXW, max((im.width for im in imgs.values()), default=0) + 2 * PAD)
    place, x, y, rowh = {}, PAD, PAD, 0
    for n in sorted(all_regions, key=lambda k: -imgs[k].height):
        w, h = imgs[n].size
        if x + w + PAD > MAXW:
            x, y, rowh = PAD, y + rowh + PAD, 0
        place[n] = (x, y); x += w + PAD; rowh = max(rowh, h)
    pageh = y + rowh + PAD
    page = Image.new("RGBA", (MAXW, pageh), (0, 0, 0, 0))
    for n, (px, py) in place.items():
        page.paste(imgs[n], (px, py))
    page.save(f"{out_dir}/{name}.png")
    al = ["", f"{name}.png", f"size: {MAXW},{pageh}", "filter: Linear,Linear", "repeat: none"]
    for n in all_regions:
        w, h = imgs[n].size; px, py = place[n]
        al += [n, "  rotate: false", f"  xy: {px}, {py}", f"  size: {w}, {h}",
               f"  orig: {w}, {h}", "  offset: 0, 0", "  index: -1"]
    open(f"{out_dir}/{name}.atlas", "w").write("\n".join(al) + "\n")

    # ---- bones / slots / skin ------------------------------------------------
    bones = [{"name": "root"}]
    for bn, b in BONES.items():
        pw = bworld(b["parent"])
        lx, ly = to_local(b["x"], b["y"], pw)
        bones.append({"name": bn, "parent": b["parent"], "x": round(lx, 2),
                      "y": round(ly, 2), "rotation": round(b["rot"] - pw[2], 2) or 0,
                      "scaleX": 1.0, "scaleY": 1.0, "length": b["length"] or 0})
    GLOW = "Layer 2" if "Layer 2" in draw_final else None
    # molten lava / fire parts → additive blend so a colour pulse reads as heat
    GLOWSET = ([GLOW] if GLOW else []) + sorted(FIRE)
    slots = []
    for n in draw_final:
        s = {"name": n, "bone": slotbone(n), "attachment": n}
        if n in GLOWSET or n in ADDITIVE_FX:
            s["blend"] = "additive"
        slots.append(s)

    def att_entry(region, host_slot):
        bw = bworld(slotbone(host_slot))
        lx, ly = to_local(norm[region]["cx"], norm[region]["cy"], bw)
        e = {"x": round(lx, 2), "y": round(ly, 2)}
        if bw[2]:
            e["rotation"] = round(-bw[2], 2)
        e.update({"width": int(norm[region]["w"]), "height": int(norm[region]["h"])})
        if region != host_slot:
            e["path"] = region
        return e

    attachments = {}
    for n in draw_final:
        entry = {n: att_entry(n, n)}
        if n in STATE_FAM:
            for st, variant in STATE_FAM[n].items():
                if st != "base":
                    entry[variant] = att_entry(variant, n)
        attachments[n] = entry
    skins = [{"name": "default", "attachments": attachments}]

    HEAD_SLOT = next((n for n in draw_final if n in STATE_FAM), None)
    winface = STATE_FAM.get(HEAD_SLOT, {}).get("win") if HEAD_SLOT else None
    blinkface = STATE_FAM.get(HEAD_SLOT, {}).get("blink") if HEAD_SLOT else None

    # ---- animations ----------------------------------------------------------
    want = set(anims or ["idle", "win", "blink", "pop"])
    animations = {}

    D = 2.8
    ib = {"body": {"scale": [{"time": 0, "x": 1, "y": 1}, {"time": 1.4, "x": 1.045, "y": 0.985}, {"time": D, "x": 1, "y": 1}],
                   "rotate": [{"time": 0, "value": 0}, {"time": 0.9, "value": 1.7}, {"time": 1.9, "value": -1.4}, {"time": D, "value": 0}]}}
    if has_head:
        ib["head"] = {"rotate": [{"time": 0, "value": 0}, {"time": 0.9, "value": 3}, {"time": 1.9, "value": -2.2}, {"time": D, "value": 0}]}
    if has_rot:
        ib["rot"] = {"rotate": [{"time": 0, "value": 0}, {"time": D / 2, "value": 180}, {"time": D, "value": 360}]}
    idle = {"bones": ib}
    # idle lava/fire: slow molten breathe (slightly irregular dim↔bright, loop closes)
    if GLOWSET:
        idle["slots"] = {n: {"rgba": [
            {"time": 0, "color": "ffffffff"}, {"time": 0.7, "color": "ffcc99ff"},
            {"time": 1.3, "color": "fff2e0ff"}, {"time": 1.9, "color": "ffc285ff"},
            {"time": 2.4, "color": "fff7ecff"}, {"time": D, "color": "ffffffff"}]} for n in GLOWSET}
    if "idle" in want:
        animations["idle"] = idle

    Wd = 0.72
    wb = {"body": {
        "scale": [{"time": 0, "x": 1, "y": 1}, {"time": 0.1, "x": 1.32, "y": 0.76}, {"time": 0.28, "x": 0.8, "y": 1.26}, {"time": 0.46, "x": 1.14, "y": 0.9}, {"time": 0.6, "x": 0.98, "y": 1.03}, {"time": Wd, "x": 1, "y": 1}],
        "rotate": [{"time": 0, "value": 0}, {"time": 0.16, "value": -10}, {"time": 0.36, "value": 10}, {"time": 0.54, "value": -4}, {"time": Wd, "value": 0}],
        "translate": [{"time": 0, "x": 0, "y": 0}, {"time": 0.28, "x": 0, "y": 22}, {"time": 0.5, "x": 0, "y": 0}, {"time": Wd, "x": 0, "y": 0}]}}
    if has_head:
        wb["head"] = {"rotate": [{"time": 0, "value": 0}, {"time": 0.3, "value": 11}, {"time": 0.52, "value": -5}, {"time": Wd, "value": 0}]}
    win = {"bones": wb, "slots": {}}
    if winface:
        win["slots"][HEAD_SLOT] = {"attachment": [{"time": 0, "name": winface}, {"time": Wd, "name": HEAD_SLOT}]}
    for n in GLOWSET:   # lava flares white-hot with a quick flicker on a win
        win["slots"][n] = {"rgba": [{"time": 0, "color": "ffffffff"}, {"time": 0.12, "color": "fff2e0ff"},
                                    {"time": 0.26, "color": "ffd9a0ff"}, {"time": 0.4, "color": "ffffffff"},
                                    {"time": Wd, "color": "ffffffff"}]}
    if not win["slots"]:
        del win["slots"]
    if "win" in want:
        animations["win"] = win

    if "blink" in want and blinkface:
        animations["blink"] = {"slots": {HEAD_SLOT: {"attachment": [
            {"time": 0, "name": HEAD_SLOT}, {"time": 0.12, "name": blinkface}, {"time": 0.22, "name": HEAD_SLOT}]}}}

    if "pop" in want:
        Pd = 0.36
        animations["pop"] = {"bones": {"body": {
            "scale": [{"time": 0, "x": 1.12, "y": 0.84}, {"time": 0.16, "x": 0.94, "y": 1.08}, {"time": Pd, "x": 1, "y": 1}],
            "translate": [{"time": 0, "x": 0, "y": 10}, {"time": 0.16, "x": 0, "y": -3}, {"time": Pd, "x": 0, "y": 0}]}}}

    # ignite — lava "catches fire": cold dark ember → red → orange → white-hot with
    # a flicker burst, ending at the bright idle baseline so it mixes back; a small
    # body scale-pop punctuates the heat surge. Only when there's a glow/lava part.
    if GLOWSET and (anims is None or "ignite" in want):
        Ig = 1.35
        glow_ignite = {"rgba": [
            {"time": 0, "color": "2e160bff"}, {"time": 0.18, "color": "7a3010ff"},
            {"time": 0.4, "color": "d9702aff"}, {"time": 0.62, "color": "ffffffff"},
            {"time": 0.76, "color": "ffd49bff"}, {"time": 0.9, "color": "ffffffff"},
            {"time": 1.05, "color": "ffe3b8ff"}, {"time": 1.18, "color": "ffffffff"},
            {"time": Ig, "color": "ffffffff"}]}
        animations["ignite"] = {"slots": {n: dict(glow_ignite) for n in GLOWSET},
                                "bones": {"body": {"scale": [
                                    {"time": 0, "x": 1, "y": 1}, {"time": 0.55, "x": 0.99, "y": 1.0},
                                    {"time": 0.64, "x": 1.07, "y": 0.95}, {"time": 0.8, "x": 0.98, "y": 1.02},
                                    {"time": Ig, "x": 1, "y": 1}]}}}

    # ---- limbs in the animations ---------------------------------------------
    # Arms only: legs stay planted and follow the hips. Rotations are signed so
    # "outward" means away from the body on either side.
    for chain, names in LIMB_CHAINS.items():
        if not chain.startswith("arm_"):
            continue
        s = OUTWARD[names[0]]
        if "idle" in animations:
            ib = animations["idle"]["bones"]
            ib[names[0]] = {"rotate": [{"time": 0, "value": 0}, {"time": 0.9, "value": 2.5 * s},
                                       {"time": 1.9, "value": -2 * s}, {"time": D, "value": 0}]}
            if len(names) > 1:
                ib[names[1]] = {"rotate": [{"time": 0, "value": 0}, {"time": 1.1, "value": 3 * s},
                                           {"time": 2.1, "value": -2.5 * s}, {"time": D, "value": 0}]}
        if "win" in animations:
            wb = animations["win"]["bones"]
            wb[names[0]] = {"rotate": [{"time": 0, "value": 0}, {"time": 0.1, "value": -6 * s},
                                       {"time": 0.3, "value": 38 * s}, {"time": 0.5, "value": 28 * s},
                                       {"time": Wd, "value": 0}]}
            if len(names) > 1:
                wb[names[1]] = {"rotate": [{"time": 0, "value": 0}, {"time": 0.34, "value": 22 * s},
                                           {"time": 0.54, "value": 12 * s}, {"time": Wd, "value": 0}]}
    if has_legs:
        # a body translate on hips moves the legs with it (jumps, landings)
        for a in animations.values():
            t = a.get("bones", {}).get("body", {}).pop("translate", None)
            if t:
                a["bones"]["hips"] = {"translate": t}

    skel = {"skeleton": {"hash": f"spine-mcp-{name}", "spine": "4.2.00",
                         "x": round(-W / 2, 2), "y": 0, "width": round(W, 2), "height": round(H, 2),
                         "images": "./", "audio": ""},
            "bones": bones, "slots": slots, "skins": skins, "animations": animations}
    open(f"{out_dir}/{name}.json", "w").write(json.dumps(skel))

    return {
        "name": name, "width": round(W), "height": round(H),
        "bones": [b["name"] for b in bones], "slots": [s["name"] for s in slots],
        "head_slot": HEAD_SLOT, "variants": list(STATE_FAM.get(HEAD_SLOT, {})) if HEAD_SLOT else [],
        "anims": list(animations),
        "limbs": LIMB_CHAINS, "warnings": warnings,
        "files": {"json": f"{out_dir}/{name}.json", "atlas": f"{out_dir}/{name}.atlas", "png": f"{out_dir}/{name}.png"},
    }
