"""core.py - virtual try-on engine for AI Outfit Studio.

Pure NumPy / Pillow / OpenCV (no PyTorch), so it is easy to test.
  * procedural clothes catalogue (drawn in code, no copyrighted images)
  * GAN garment post-processing (turn a 28x28 GAN sample into a tinted garment)
  * face detection, skin-tone / colour-season analysis, suitability scoring
  * fitting a garment onto a person's photo
"""
import os
import zlib
import functools
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
CW, CH = 800, 1000            # garment canvas
SS = 2                        # supersampling for drawing
ANCHOR = (400, 150)           # neck-centre of every garment canvas

COLORS = {
    "Black": (28, 28, 32), "White": (245, 245, 242), "Charcoal": (70, 74, 82),
    "Navy": (28, 44, 92), "Royal Blue": (40, 80, 200), "Sky Blue": (140, 190, 235),
    "Teal": (20, 128, 128), "Emerald": (10, 130, 90), "Olive": (105, 115, 50),
    "Mustard": (212, 165, 30), "Orange": (235, 120, 30), "Coral": (255, 115, 95),
    "Red": (200, 25, 45), "Maroon": (115, 25, 45), "Pink": (240, 150, 180),
    "Lavender": (180, 160, 220), "Purple": (110, 50, 150), "Beige": (225, 205, 170),
    "Brown": (110, 70, 45), "Terracotta": (195, 95, 62),
}
PATTERNS = ["Solid", "Stripes", "Checks", "Dots"]

GARMENTS = {   # key -> label
    "tshirt": "Crew T-shirt", "vneck": "V-neck tee", "tank": "Tank top",
    "shirt": "Collared shirt", "hoodie": "Hoodie", "jacket": "Blazer / jacket",
    "dress": "Dress", "kurta": "Kurta",
}
NECK_DEPTH = {"tshirt": 88, "vneck": 118, "tank": 155, "shirt": 95, "hoodie": 65,
              "jacket": 320, "dress": 82, "kurta": 55}
GAN_CLASSES = ["T-shirt", "Pullover", "Dress", "Coat", "Shirt"]
GAN_SHOULDER_FRAC = [0.78, 0.66, 0.55, 0.68, 0.72]   # shoulder width / bbox width


# ----------------------------------------------------------------- helpers
def _bez(p0, p1, p2, n=24):
    t = np.linspace(0, 1, n)
    return [((1 - a) ** 2 * p0[0] + 2 * (1 - a) * a * p1[0] + a * a * p2[0],
             (1 - a) ** 2 * p0[1] + 2 * (1 - a) * a * p1[1] + a * a * p2[1]) for a in t]

def _mirror(pts):
    return [(CW - x, y) for x, y in pts]

def _seed(*a):
    return zlib.crc32(repr(a).encode()) & 0xFFFFFFFF

def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

def _dark(c, k=0.62):
    return tuple(int(v * k) for v in c)

def _light(c):
    return tuple(int(min(255, v * 1.2 + 35)) for v in c)


# --------------------------------------------------------- garment geometry
def _geometry(kind):
    """Return dict(outline, holes, shoulder, neckline, ops)."""
    holes, ops = [], []
    if kind in ("tshirt", "vneck", "tank"):
        if kind == "tank":
            left = [(332, 122), (222, 185), (240, 300), (240, 730)]
            neck = _bez((468, 122), (400, 305), (332, 122))
            sh = (222, 578)
        else:
            left = [(335, 135), (200, 172), (108, 312), (186, 358), (236, 300), (236, 730)]
            sh = (200, 600)
            neck = (_bez((465, 135), (400, 238), (335, 135)) if kind == "tshirt"
                    else [(465, 135), (400, 268), (335, 135)])
        outline = left + _mirror(left[::-1]) + neck[1:-1]
        ops.append(("line", neck, 14, "dark", 150))
        ops.append(("line", [(236, 705), (564, 705)], 4, "dark", 110))
        if kind != "tank":
            ops.append(("line", [(236, 300), (200, 172)], 3, "dark", 90))
            ops.append(("line", [(564, 300), (600, 172)], 3, "dark", 90))
        return dict(outline=outline, holes=holes, shoulder=sh,
                    neckline="v" if kind == "vneck" else "crew", ops=ops)

    if kind == "shirt":
        left = [(335, 135), (200, 172), (108, 312), (186, 358), (236, 300), (236, 730)]
        neck = [(465, 135), (400, 245), (335, 135)]
        outline = left + _mirror(left[::-1]) + neck[1:-1]
        ops += [("poly", [(335, 126), (405, 238), (348, 266), (312, 168)], "light", "dark"),
                ("poly", _mirror([(335, 126), (405, 238), (348, 266), (312, 168)]), "light", "dark"),
                ("line", [(400, 245), (400, 725)], 4, "dark", 120),
                ("line", [(236, 705), (564, 705)], 4, "dark", 110),
                ("rect", (436, 340, 524, 420), 4, "dark", 130)]
        for i in range(5):
            ops.append(("dot", (400, 300 + i * 85), 7, "dark", 200))
        return dict(outline=outline, holes=holes, shoulder=(200, 600), neckline="collar", ops=ops)

    if kind in ("hoodie", "jacket"):
        left = [(335, 135), (200, 172), (118, 520), (98, 820), (178, 836), (196, 530),
                (238, 330), (234, 830)]
        if kind == "hoodie":
            neck = _bez((465, 135), (400, 215), (335, 135))
            outline = left + _mirror(left[::-1]) + neck[1:-1]
            ops += [("line", _bez((312, 138), (400, 330), (488, 138)), 30, "dark", 170),
                    ("line", [(378, 240), (374, 340)], 5, "white", 220),
                    ("line", [(422, 240), (426, 340)], 5, "white", 220),
                    ("poly", [(292, 620), (508, 620), (536, 760), (264, 760)], None, "dark"),
                    ("line", [(236, 808), (564, 808)], 8, "dark", 150),
                    ("line", [(98, 805), (178, 821)], 8, "dark", 150),
                    ("line", [(702, 805), (622, 821)], 8, "dark", 150)]
            return dict(outline=outline, holes=holes, shoulder=(200, 600),
                        neckline="hood", ops=ops)
        # jacket: deep V opening with a white tee beneath
        vopen = [(465, 135), (400, 470), (335, 135)]
        outline = left + _mirror(left[::-1]) + vopen[1:-1]
        ops += [("poly", [(335, 135), (400, 470), (302, 300), (312, 168)], "dark", "dark"),
                ("poly", _mirror([(335, 135), (400, 470), (302, 300), (312, 168)]), "dark", "dark"),
                ("line", [(400, 470), (400, 830)], 4, "dark", 140),
                ("dot", (400, 560), 9, "dark", 220), ("dot", (400, 680), 9, "dark", 220),
                ("rect", (270, 700, 350, 740), 4, "dark", 130),
                ("rect", (450, 700, 530, 740), 4, "dark", 130),
                ("line", [(234, 808), (566, 808)], 5, "dark", 120)]
        return dict(outline=outline, holes=holes, shoulder=(200, 600), neckline="v",
                    ops=ops, inner_tee=True)

    if kind == "dress":
        left = [(335, 135), (228, 168), (246, 310), (272, 520), (150, 930)]
        neck = _bez((465, 135), (400, 232), (335, 135))
        outline = left + _mirror(left[::-1]) + neck[1:-1]
        ops += [("line", neck, 12, "dark", 150),
                ("poly", [(268, 505), (532, 505), (537, 548), (263, 548)], "dark", "dark"),
                ("line", [(150, 925), (650, 925)], 5, "dark", 110)]
        return dict(outline=outline, holes=holes, shoulder=(228, 572), neckline="crew", ops=ops)

    if kind == "kurta":
        left = [(350, 135), (210, 172), (125, 380), (112, 650), (192, 660), (236, 340),
                (228, 700), (205, 990)]
        neck = _bez((450, 135), (400, 205), (350, 135))
        outline = left + _mirror(left[::-1]) + neck[1:-1]
        ops += [("line", neck, 12, "gold", 240),
                ("line", [(400, 205), (400, 480)], 10, "gold", 240),
                ("line", [(112, 640), (192, 650)], 14, "gold", 240),
                ("line", [(688, 640), (608, 650)], 14, "gold", 240),
                ("rect", (210, 935, 590, 970), 4, "gold", 230),
                ("dot", (400, 280), 6, "gold", 255), ("dot", (400, 360), 6, "gold", 255),
                ("dot", (400, 440), 6, "gold", 255)]
        for i in range(9):
            x = 232 + i * 42
            ops.append(("poly", [(x, 952), (x + 12, 940), (x + 24, 952), (x + 12, 964)], "gold", None))
        return dict(outline=outline, holes=holes, shoulder=(210, 590), neckline="crew", ops=ops)
    raise ValueError(kind)


def _resolve(tok, base):
    if tok is None:
        return None
    if isinstance(tok, tuple):
        return tok
    return {"dark": _dark(base), "light": _light(base), "white": (250, 250, 250),
            "gold": (214, 176, 70)}[tok]


# ------------------------------------------------------------ fabric render
def _pattern(size, base, pattern):
    w, h = size
    base = np.array(base, np.float32)
    img = np.ones((h, w, 3), np.float32) * base
    sec = np.clip(base * 0.6, 0, 255) if _lum(base) > 110 else np.clip(base + (255 - base) * 0.5, 0, 255)
    yy, xx = np.mgrid[0:h, 0:w]
    if pattern == "Stripes":
        m = ((yy % 46) < 17)[..., None]
        img = np.where(m, sec, img)
    elif pattern == "Checks":
        for m in (((xx % 70) < 26), ((yy % 70) < 26)):
            m = m[..., None] * 0.5
            img = img * (1 - m) + sec * m
    elif pattern == "Dots":
        cx = ((xx + 23 * ((yy // 46) % 2)) % 46) - 23
        cy = (yy % 46) - 23
        m = ((cx * cx + cy * cy) < 64)[..., None]
        img = np.where(m, sec, img)
    return img


def _shade(rgb, alpha, seed):
    h, w = alpha.shape
    rng = np.random.default_rng(seed)
    folds = cv2.resize(rng.normal(0, 1, (10, 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    folds = cv2.GaussianBlur(folds, (0, 0), 25)
    folds /= (np.abs(folds).max() + 1e-6)
    light = np.linspace(1.06, 0.92, w, dtype=np.float32)[None, :]
    inner = cv2.GaussianBlur(alpha, (0, 0), 16)
    edge = 0.72 + 0.28 * np.clip((inner - 0.3) / 0.7, 0, 1)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.8) * 0.03
    s = (1 + 0.07 * folds) * light * edge + tex
    return np.clip(rgb * s[..., None], 0, 255)


def _to_rgba(rgb, alpha):
    out = np.dstack([rgb, alpha * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


@functools.lru_cache(maxsize=256)
def _proc_cached(kind, rgb, pattern):
    geo = _geometry(kind)
    S = lambda pts: [(x * SS, y * SS) for x, y in pts]
    m = Image.new("L", (CW * SS, CH * SS), 0)
    d = ImageDraw.Draw(m)
    d.polygon(S(geo["outline"]), fill=255)
    alpha = np.asarray(m.resize((CW, CH), Image.LANCZOS), np.float32) / 255
    fabric = _shade(_pattern((CW, CH), rgb, pattern), alpha, _seed(kind, rgb, pattern))
    base = _to_rgba(fabric, alpha)

    det = Image.new("RGBA", (CW * SS, CH * SS), (0, 0, 0, 0))
    dd = ImageDraw.Draw(det)
    for op in geo["ops"]:
        k = op[0]
        if k == "line":
            _, pts, wd, col, a = op
            dd.line(S(pts), fill=_resolve(col, rgb) + (a,), width=wd * SS, joint="curve")
        elif k == "poly":
            _, pts, fill, outl = op
            f = _resolve(fill, rgb)
            dd.polygon(S(pts), fill=(f + (235,)) if f else None)
            if outl:
                dd.line(S(pts + [pts[0]]), fill=_resolve(outl, rgb) + (200,), width=3 * SS)
        elif k == "rect":
            _, (x0, y0, x1, y1), wd, col, a = op
            dd.rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], outline=_resolve(col, rgb) + (a,), width=wd * SS)
        elif k == "dot":
            _, (x, y), r, col, a = op
            dd.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], fill=_resolve(col, rgb) + (a,))
    det = _clip_detail(det.resize((CW, CH), Image.LANCZOS), alpha)
    out = Image.alpha_composite(base, det)
    if geo.get("inner_tee"):                      # white tee visible through the jacket's V opening
        tee = Image.new("L", (CW * SS, CH * SS), 0)
        ImageDraw.Draw(tee).polygon(S([(338, 140), (462, 140), (520, 300), (520, 830), (280, 830), (280, 300)]), fill=255)
        ta = np.asarray(tee.resize((CW, CH), Image.LANCZOS), np.float32) / 255
        tf = _shade(np.ones((CH, CW, 3), np.float32) * 238, ta, 7)
        out = Image.alpha_composite(_to_rgba(tf, ta), out)
    meta = dict(shoulder_px=geo["shoulder"][1] - geo["shoulder"][0], anchor=ANCHOR,
                neckline=geo["neckline"], neck_depth=NECK_DEPTH[kind])
    return out, meta


def _clip_detail(det, alpha):
    a = np.asarray(det, np.uint8).copy()
    a[..., 3] = (a[..., 3] * alpha).astype(np.uint8)
    return Image.fromarray(a, "RGBA")


def proc_garment(kind, rgb, pattern="Solid"):
    img, meta = _proc_cached(kind, tuple(int(v) for v in rgb), pattern)
    return img.copy(), dict(meta)


def gan_garment(sample28, cls, rgb, pattern="Solid"):
    """sample28: float array 28x28 in [-1, 1] from the GAN generator."""
    g = (np.clip(sample28, -1, 1) + 1) / 2
    up = cv2.resize(g.astype(np.float32), (560, 560), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 6.0)
    mk = (up > 0.14).astype(np.uint8)
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mk, 8)
    if n > 1:                                         # keep the largest blob, fill holes
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mk = (lab == big).astype(np.uint8)
        ff = mk.copy()
        cv2.floodFill(ff, np.zeros((562, 562), np.uint8), (0, 0), 2)
        mk[ff != 2] = 1
    mk = (cv2.GaussianBlur(mk.astype(np.float32), (0, 0), 9) > 0.5).astype(np.uint8)   # smooth contour
    alpha = cv2.GaussianBlur(mk.astype(np.float32), (0, 0), 1.6)
    ys, xs = np.nonzero(mk)
    if len(xs) == 0:
        raise ValueError("empty GAN sample")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    shade = np.clip(up / (np.percentile(up[mk > 0], 90) + 1e-6), 0.35, 1.15)
    shade = 0.55 + 0.55 * shade
    fabric = _pattern((560, 560), rgb, pattern) * shade[..., None]
    fabric = _shade(fabric, alpha, _seed("gan", cls, rgb))
    img = _to_rgba(fabric, alpha)
    bw = x1 - x0 + 1
    meta = dict(shoulder_px=float(bw * GAN_SHOULDER_FRAC[int(cls)]),
                anchor=(float((x0 + x1) / 2), float(y0 + 0.05 * (y1 - y0))), neckline="crew", neck_depth=0)
    return img, meta


def flatten(img, bg=(255, 255, 255)):
    canvas = Image.new("RGB", img.size, bg)
    canvas.paste(img, (0, 0), img)
    return canvas


# ------------------------------------------------------------ face + colour
_CASCADE = None

def load_cascade():
    global _CASCADE
    if _CASCADE is not None:
        return _CASCADE
    name = "haarcascade_frontalface_default.xml"
    cands = [os.path.join(HERE, name)]
    try:
        cands.append(cv2.data.haarcascades + name)
    except Exception:
        pass
    cands.append(os.path.join(os.path.dirname(cv2.__file__), "data", name))
    for p in cands:
        if os.path.exists(p):
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                _CASCADE = c
                return c
    _CASCADE = False
    return False


def prepare_photo(img, max_side=900):
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        k = max_side / max(img.size)
        img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
    return img


def detect_face(img):
    """Return ((x, y, w, h), found). Falls back to a guess if no face is found."""
    arr = np.array(img.convert("RGB"))
    gray = cv2.equalizeHist(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY))
    c = load_cascade()
    found = []
    if c:
        for sf, mn in ((1.1, 5), (1.05, 3)):
            found = c.detectMultiScale(gray, sf, mn, minSize=(max(40, img.width // 12),) * 2)
            if len(found):
                break
    if len(found):
        x, y, w, h = max(found, key=lambda f: f[2] * f[3])
        return (int(x), int(y), int(w), int(h)), True
    W, H = img.size
    w = int(0.34 * min(W, H * 0.8))
    return (int(W / 2 - w / 2), int(0.12 * H), w, w), False


def _lab(rgb):
    return cv2.cvtColor(np.float32([[rgb]]) / 255.0, cv2.COLOR_RGB2LAB)[0, 0]

SEASONS = {
    "Spring": [(255, 127, 80), (255, 203, 164), (0, 180, 170), (255, 200, 40), (120, 190, 60),
               (255, 140, 160), (200, 150, 90), (255, 250, 235)],
    "Summer": [(150, 185, 225), (180, 160, 220), (230, 140, 170), (90, 160, 160), (110, 130, 160),
               (180, 130, 160), (60, 80, 130), (170, 175, 185)],
    "Autumn": [(200, 95, 60), (205, 160, 30), (110, 120, 50), (170, 70, 30), (30, 100, 100),
               (215, 110, 30), (100, 60, 40), (245, 230, 200), (130, 40, 40)],
    "Winter": [(200, 20, 40), (30, 70, 190), (0, 130, 90), (20, 20, 25), (250, 250, 250),
               (200, 30, 130), (55, 60, 70), (90, 40, 140)],
}
UNIVERSAL = [(28, 44, 92), (70, 74, 82), (245, 245, 242), (28, 28, 32)]   # navy, charcoal, white, black
SEASON_BLURB = {
    "Spring": "warm and bright - clear, fresh, sunny colours",
    "Summer": "cool and soft - gentle, dusty, slightly muted colours",
    "Autumn": "warm and deep - earthy, rich, spicy colours",
    "Winter": "cool and strong - crisp, saturated, high-contrast colours",
}


def analyze_person(img, face):
    x, y, w, h = face
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]
    patches = []
    for fx0, fx1 in ((0.12, 0.38), (0.62, 0.88)):
        x0, x1 = int(x + fx0 * w), int(x + fx1 * w)
        y0, y1 = int(y + 0.42 * h), int(y + 0.74 * h)
        patches.append(arr[max(0, y0):min(H, y1), max(0, x0):min(W, x1)].reshape(-1, 3))
    px = np.concatenate([p for p in patches if len(p)]) if any(len(p) for p in patches) else arr.reshape(-1, 3)
    ycc = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    ok = (ycc[:, 1] > 133) & (ycc[:, 1] < 173) & (ycc[:, 2] > 77) & (ycc[:, 2] < 127)
    if ok.sum() > 30:
        px = px[ok]
    skin = tuple(int(v) for v in np.median(px, axis=0))
    L, a, b = _lab(skin)
    hue = float(np.degrees(np.arctan2(b, a)))
    undertone = "warm" if hue >= 57 else ("cool" if hue <= 48 else "neutral")
    depth = "light" if L > 68 else ("medium" if L > 52 else "deep")
    hy0, hy1 = max(0, int(y - 0.35 * h)), int(y + 0.02 * h)
    contrast = "medium"
    if hy1 - hy0 > 6:
        hair = arr[hy0:hy1, int(x + 0.2 * w):int(x + 0.8 * w)].reshape(-1, 3)
        if len(hair):
            hl = _lab(tuple(np.median(hair, axis=0)))[0]
            d = abs(hl - L)
            contrast = "high" if d > 35 else ("medium" if d > 20 else "low")
    if undertone == "warm":
        season = "Spring" if depth == "light" else "Autumn"
    elif undertone == "cool":
        season = "Summer" if depth == "light" else "Winter"
    else:
        season = "Summer" if depth == "light" else ("Winter" if contrast == "high" else "Autumn")
    return dict(skin=skin, skin_L=float(L), hue=hue, undertone=undertone, depth=depth,
                contrast=contrast, season=season)


def score_color(an, rgb, neckline="crew"):
    gl, ga, gb = _lab(rgb)
    # 1) palette match
    dists = [float(np.linalg.norm(_lab(c) - np.array([gl, ga, gb]))) for c in SEASONS[an["season"]]]
    pal = 100 * float(np.exp(-min(dists) / 45.0))
    neutral = [float(np.linalg.norm(_lab(c) - np.array([gl, ga, gb]))) for c in UNIVERSAL]
    pal = max(pal, 70 * float(np.exp(-min(neutral) / 25.0)))
    # 2) contrast with skin
    dL = abs(gl - an["skin_L"])
    if an["contrast"] == "high":
        con = min(dL / 30.0, 1.0) * 100
    elif an["contrast"] == "low":
        con = max(0.0, 100 - abs(dL - 22) * 3.2)
    else:
        con = max(0.0, 100 - abs(dL - 30) * 2.2)
    # 3) undertone / temperature
    chroma = float(np.hypot(ga, gb))
    t = float(np.cos(np.radians(np.degrees(np.arctan2(gb, ga)) - 65))) * min(chroma / 35.0, 1.0)
    if an["undertone"] == "warm":
        und = 50 + 50 * t
    elif an["undertone"] == "cool":
        und = 50 - 50 * t
    else:
        und = 70 + 10 * (1 - abs(t))
    raw = 0.5 * pal + 0.25 * con + 0.25 * und
    total = 100 * (np.clip(raw, 0, 100) / 100) ** 0.8
    return dict(total=float(np.clip(total, 0, 100)), palette=pal, contrast=float(con), undertone=float(und))


def verdict(total):
    if total >= 80:
        return "Excellent match", "✨"
    if total >= 65:
        return "Great match", "👍"
    if total >= 50:
        return "Decent - it works", "🙂"
    return "Not your best colour", "🤔"


def tips(an, sc, rgb):
    out = [f"Your colouring reads as **{an['undertone']} undertone, {an['depth']} skin, {an['contrast']} contrast** "
           f"→ colour family: **{an['season']}** ({SEASON_BLURB[an['season']]})."]
    if sc["palette"] < 45:
        out.append(f"This shade sits outside the {an['season']} palette - try one of the swatches below, "
                   f"or wear it away from your face (as trousers/jacket).")
    if sc["contrast"] < 45:
        out.append("The contrast with your skin is not ideal for your colouring - " +
                   ("pick something bolder." if an["contrast"] == "high" else "pick something softer/closer in tone."))
    if sc["undertone"] < 45:
        out.append(f"The colour's temperature clashes a little with your {an['undertone']} undertone.")
    if sc["total"] >= 80:
        out.append("This colour brightens your complexion - a safe choice for photos and interviews.")
    return out


def rank_catalog(an, kinds=None, top=8):
    """Best colours for this person, each paired with a different garment type."""
    kinds = list(kinds or GARMENTS)
    cols = sorted(((score_color(an, rgb)["total"], name, rgb) for name, rgb in COLORS.items()), reverse=True)
    return [(tot, kinds[i % len(kinds)], name, rgb) for i, (tot, name, rgb) in enumerate(cols[:top])]


# ----------------------------------------------------------------- fitting
def fit_garment(person, garment, meta, face, scale=1.0, dx=0.0, dy=0.0, opacity=1.0, skin=None):
    person = person.convert("RGB")
    W, H = person.size
    fx, fy, fw, fh = face
    cx = fx + fw / 2 + dx * fw
    neck_y = fy + fh * 1.18 + dy * fh
    shoulder_w = 2.8 * fw * scale
    s = shoulder_w / meta["shoulder_px"]
    gw, gh = max(1, int(garment.width * s)), max(1, int(garment.height * s))
    g = garment.resize((gw, gh), Image.LANCZOS)
    ox, oy = int(cx - meta["anchor"][0] * s), int(neck_y - meta["anchor"][1] * s)

    # match brightness to the photo
    box = (max(0, fx), max(0, fy), min(W, fx + fw), min(H, fy + fh))
    region = person.crop(box) if box[2] > box[0] and box[3] > box[1] else person
    face_gray = np.asarray(region.convert("L"), np.float32).mean()
    k = float(np.clip(face_gray / 150.0, 0.72, 1.15)) ** 0.8
    ga = np.asarray(g, np.float32).copy()
    ga[..., :3] = np.clip(ga[..., :3] * k, 0, 255)
    ga[..., 3] *= opacity
    g = Image.fromarray(ga.astype(np.uint8), "RGBA")

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.paste(g, (ox, oy))
    # soft contact shadow
    sh_a = layer.getchannel("A").filter(ImageFilter.GaussianBlur(max(2, W // 70)))
    sh_a = ImageChops.offset(sh_a, 0, max(2, H // 90)).point(lambda v: int(v * 0.35))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    shadow.putalpha(sh_a)

    base = person.convert("RGBA")
    if skin is not None and meta.get("neck_depth", 0) > 0:      # bare neck / chest inside the neckline
        bottom = neck_y + meta["neck_depth"] * s
        top = neck_y - 0.02 * fh
        poly = [(cx - 0.26 * fw, top), (cx + 0.26 * fw, top),
                (cx + 0.50 * fw, bottom), (cx - 0.50 * fw, bottom)]
        patch = Image.new("L", (W, H), 0)
        ImageDraw.Draw(patch).polygon(poly, fill=255)
        patch = patch.filter(ImageFilter.GaussianBlur(max(2, fw * 0.04)))
        tone = tuple(int(v * 0.95) for v in skin)
        base = Image.composite(Image.new("RGBA", (W, H), tone + (255,)), base, patch)
    base = Image.alpha_composite(base, shadow)
    out = Image.alpha_composite(base, layer).convert("RGB")

    # keep the real face / hair on top
    prot = Image.new("L", (W, H), 0)
    ImageDraw.Draw(prot).ellipse([fx - 0.18 * fw, fy - 0.55 * fh, fx + 1.18 * fw, fy + 1.10 * fh], fill=255)
    prot = prot.filter(ImageFilter.GaussianBlur(max(2, fw * 0.05)))
    return Image.composite(person, out, prot)
