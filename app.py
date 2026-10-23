"""AI Outfit Studio - virtual try-on + colour-suitability + GAN garment designer (Streamlit)."""
import hashlib
import io
import json
import os
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

import core

st.set_page_config(page_title="AI Outfit Studio", page_icon="👗", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    CFG = json.load(open(os.path.join(HERE, "config.json")))
except Exception:
    CFG = {"model_repo": ""}
try:
    HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 12% 0%, #2a1744 0%, #0b0d1a 58%);}
h1.hero {font-size: 3rem; font-weight: 800; margin-bottom: 0;
  background: linear-gradient(90deg,#ff7eb3,#a78bfa,#34d399);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.sub {color:#a7a9be; font-size:1.1rem; margin-bottom:1rem;}
.card {background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.10);
  border-radius:16px; padding:14px 16px; margin-bottom:10px;}
.big {font-size:3.2rem; font-weight:800; line-height:1;}
.big span {font-size:1.1rem; color:#a7a9be; font-weight:500;}
.bar {height:9px; border-radius:6px; background:rgba(255,255,255,.12); overflow:hidden; margin:3px 0 10px;}
.bar > div {height:100%; border-radius:6px; background:linear-gradient(90deg,#ff7eb3,#a78bfa);}
.sw {display:inline-block; width:34px; height:34px; border-radius:50%; margin:3px 4px 0 0;
  border:2px solid rgba(255,255,255,.35);}
div.stButton > button {border-radius:12px; font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ GAN
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:
    TORCH_OK = False

if TORCH_OK:
    class Generator(nn.Module):
        def __init__(self, z_dim=100, n_classes=5):
            super().__init__()
            self.embed = nn.Embedding(n_classes, 50)
            self.fc = nn.Sequential(nn.Linear(z_dim + 50, 128 * 7 * 7),
                                    nn.BatchNorm1d(128 * 7 * 7), nn.ReLU(True))
            self.net = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

        def forward(self, z, y):
            h = self.fc(torch.cat([z, self.embed(y)], 1)).view(-1, 128, 7, 7)
            return self.net(h)


@st.cache_resource(show_spinner="Loading the garment GAN from the Hugging Face Hub...")
def load_gan():
    if not TORCH_OK or not CFG.get("model_repo"):
        return None
    try:
        from huggingface_hub import hf_hub_download
        repo = CFG["model_repo"]
        cfg = json.load(open(hf_hub_download(repo, "config.json", token=HF_TOKEN)))
        g = Generator(cfg["z_dim"], cfg["n_classes"])
        g.load_state_dict(torch.load(hf_hub_download(repo, "generator.pt", token=HF_TOKEN), map_location="cpu"))
        g.eval()
        return g, cfg["z_dim"]
    except Exception as e:                       # keep the rest of the app working
        st.session_state["gan_error"] = str(e)[:200]
        return None


def gan_sample(cls, seed):
    g, zd = GAN
    gen = torch.Generator().manual_seed(int(seed))
    z = torch.randn(1, zd, generator=gen)
    with torch.no_grad():
        return g(z, torch.tensor([int(cls)]))[0, 0].numpy()


GAN = load_gan()


def build_garment(spec):
    if spec[0] == "proc":
        _, t, rgb, pat = spec
        return core.proc_garment(t, rgb, pat)
    _, cls, rgb, pat, seed = spec
    return core.gan_garment(gan_sample(cls, seed), cls, rgb, pat)


def describe(spec):
    if spec[0] == "proc":
        return f"{core.GARMENTS[spec[1]]} · {spec[3]}"
    return f"GAN {core.GAN_CLASSES[spec[1]]} #{spec[4]} · {spec[3]}"


def color_name(rgb):
    for n, c in core.COLORS.items():
        if tuple(c) == tuple(rgb):
            return n
    return "custom colour"


# -------------------------------------------------------------- callbacks
def wear_catalog(kind, cname):
    st.session_state["ptype"], st.session_state["pcolor"] = kind, cname
    st.session_state["ppat"] = "Solid"
    st.session_state.pop("override", None)


def wear_override(spec):
    st.session_state["override"] = spec


def clear_override():
    st.session_state.pop("override", None)


def shuffle_seed():
    st.session_state["gseed"] = int(np.random.randint(0, 10 ** 6))


# ------------------------------------------------------------------ header
st.markdown('<h1 class="hero">👗 AI Outfit Studio</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload your photo, try clothes on, and see which colours actually suit you - '
            'plus garments designed by a GAN.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Your photo")
    mode = st.radio("Source", ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")
    photo = (st.file_uploader("Front-facing, shoulders visible", type=["jpg", "jpeg", "png"])
             if mode == "Upload" else st.camera_input("Take a selfie"))
    st.caption("🔒 Processed in memory for this session only - not stored.")
    st.header("2 · Fit")
    scale = st.slider("Garment size", 0.7, 1.4, 1.0, 0.02)
    dx = st.slider("Move left / right", -0.6, 0.6, 0.0, 0.02)
    dy = st.slider("Move up / down", -0.5, 0.8, 0.0, 0.02)
    opacity = st.slider("Opacity", 0.5, 1.0, 1.0, 0.05)
    with st.expander("Face box (use if auto-detect is off)"):
        manual = st.checkbox("Set face box manually")
        mcx = st.slider("Face centre X", 0.0, 1.0, 0.5, 0.01)
        mty = st.slider("Face top Y", 0.0, 1.0, 0.15, 0.01)
        msz = st.slider("Face width", 0.05, 0.8, 0.3, 0.01)


@st.cache_data(show_spinner="Analysing your photo...")
def analyse(raw):
    img = core.prepare_photo(Image.open(io.BytesIO(raw)))
    face, found = core.detect_face(img)
    return img, face, found


person = face = an = None
found = False
photo_key = ""
if photo is not None:
    raw = photo.getvalue()
    photo_key = hashlib.md5(raw).hexdigest()
    person, face, found = analyse(raw)
    if manual:
        W, H = person.size
        w = int(msz * W)
        face, found = (int(mcx * W - w / 2), int(mty * H), w, int(w * 1.2)), True
    an = core.analyze_person(person, face)

# current garment
st.session_state.setdefault("ptype", "tshirt")
st.session_state.setdefault("pcolor", "Navy")
st.session_state.setdefault("ppat", "Solid")
override = st.session_state.get("override")
if override:
    spec = override
else:
    spec = ("proc", st.session_state.get("ptype", "tshirt"),
            core.COLORS[st.session_state.get("pcolor", "Navy")] if st.session_state.get("pcolor", "Navy") != "Custom"
            else tuple(int(st.session_state.get("pcustom", "#1c2c5c").lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)),
            st.session_state.get("ppat", "Solid"))


@st.cache_data(show_spinner=False)
def render_fit(pkey, kind, rgb, pattern, face_, scale_, dx_, dy_, op_, _img, _skin):
    g, meta = core.proc_garment(kind, rgb, pattern)
    return core.fit_garment(_img, g, meta, face_, scale_, dx_, dy_, op_, skin=_skin)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["👗 Try-on studio", "⭐ Best for me", "🤖 GAN designer", "📸 Photo-real (diffusion)", "ℹ️ About"])

# =============================================================== TAB 1
with tab1:
    left, right = st.columns([1, 1.7])
    with left:
        st.subheader("Choose a garment")
        if override:
            st.info(f"Wearing **{describe(override)}**")
            st.button("↩ Back to the catalogue", on_click=clear_override)
        else:
            st.selectbox("Garment", list(core.GARMENTS), format_func=core.GARMENTS.get, key="ptype")
            st.selectbox("Colour", list(core.COLORS) + ["Custom"], key="pcolor")
            if st.session_state.get("pcolor") == "Custom":
                st.color_picker("Pick any colour", "#1c2c5c", key="pcustom")
            st.radio("Pattern", core.PATTERNS, horizontal=True, key="ppat")
        st.caption("Tip: a plain top in the photo works best - the new garment is drawn over it.")

    with right:
        if person is None:
            gprev, _ = build_garment(spec)
            st.image(core.flatten(gprev, (30, 30, 48)).resize((360, 450)), caption="Preview - upload a photo in the sidebar to try it on")
        else:
            if not found:
                st.warning("No face detected - using a best guess. Open **Face box** in the sidebar to place it by hand.")
            garment, meta = build_garment(spec)
            result = core.fit_garment(person, garment, meta, face, scale, dx, dy, opacity, skin=an["skin"])
            c1, c2 = st.columns(2)
            c1.image(person, caption="You", use_container_width=True)
            c2.image(result, caption=describe(spec), use_container_width=True)
            buf = io.BytesIO()
            result.save(buf, "PNG")
            st.download_button("⬇ Download this look", buf.getvalue(), "my_look.png", "image/png")

            rgb = spec[2]
            sc = core.score_color(an, rgb, meta["neckline"])
            label, emoji = core.verdict(sc["total"])
            st.markdown(f"""<div class="card"><div class="big">{sc['total']:.0f}<span> / 100</span></div>
<div style="font-size:1.25rem;margin:4px 0 10px">{emoji} <b>{label}</b> - {color_name(rgb)}</div>
<div>Colour palette match</div><div class="bar"><div style="width:{sc['palette']:.0f}%"></div></div>
<div>Contrast with your skin</div><div class="bar"><div style="width:{sc['contrast']:.0f}%"></div></div>
<div>Undertone harmony</div><div class="bar"><div style="width:{sc['undertone']:.0f}%"></div></div></div>""",
                        unsafe_allow_html=True)
            for t in core.tips(an, sc, rgb):
                st.markdown("• " + t)
            sw = "".join(f'<span class="sw" style="background:rgb{c}"></span>' for c in core.SEASONS[an["season"]])
            st.markdown(f"**Your best colours ({an['season']}):**<br>{sw}", unsafe_allow_html=True)
            st.caption("Colour guidance is a fashion heuristic from your photo's skin tone and lighting, "
                       "not a scientific verdict - trust your own eye too.")

# =============================================================== TAB 2
with tab2:
    if person is None:
        st.info("Upload a photo in the sidebar to see outfits ranked for you.")
    else:
        st.subheader(f"Top colours for your {an['season']} colouring")
        ranked = core.rank_catalog(an, top=8)
        cols = st.columns(4)
        for i, (tot, kind, cname, rgb) in enumerate(ranked):
            with cols[i % 4]:
                img = render_fit(photo_key, kind, tuple(rgb), "Solid", face, scale, dx, dy, opacity, person, an["skin"])
                st.image(img.resize((300, int(300 * img.height / img.width))), use_container_width=True)
                st.markdown(f"**{cname} {core.GARMENTS[kind]}**  \n{tot:.0f}/100")
                st.button("Wear this", key=f"wear{i}", on_click=wear_catalog, args=(kind, cname))
        st.caption("After pressing **Wear this**, open the Try-on studio tab.")

# =============================================================== TAB 3
with tab3:
    st.subheader("Garments dreamed up by a GAN")
    if GAN is None:
        st.warning("The garment GAN could not be loaded (the catalogue and try-on still work). "
                   + st.session_state.get("gan_error", ""))
    else:
        a, b, c, d = st.columns(4)
        gcls = a.selectbox("Type", list(range(len(core.GAN_CLASSES))), format_func=lambda i: core.GAN_CLASSES[i])
        gcol = b.selectbox("Colour", list(core.COLORS), index=3, key="gcol")
        gpat = c.selectbox("Pattern", core.PATTERNS, key="gpat")
        gseed = d.number_input("Seed", 0, 10 ** 6, 7, key="gseed")
        st.button("🎲 Shuffle", on_click=shuffle_seed)
        cols = st.columns(4)
        for i in range(8):
            sd = int(gseed) + i
            try:
                g, _ = core.gan_garment(gan_sample(gcls, sd), gcls, core.COLORS[gcol], gpat)
            except Exception:
                continue
            with cols[i % 4]:
                st.image(core.flatten(g, (36, 36, 54)).resize((280, 280)), use_container_width=True)
                st.button(f"Try design #{sd}", key=f"gan{i}", on_click=wear_override,
                          args=(("gan", gcls, tuple(core.COLORS[gcol]), gpat, sd),))
        st.caption("These are 28×28 GAN samples enlarged and tinted - think concept sketches. "
                   "Press **Try design**, then open the Try-on studio tab.")

# =============================================================== TAB 4
with tab4:
    st.subheader("Photo-realistic try-on (diffusion model)")
    st.write("The other tabs run on this app's own small CPU. For a photo-realistic result, this tab sends your "
             "photo and the selected garment to the public **IDM-VTON** Space on Hugging Face (a diffusion model on "
             "a shared GPU). It can be slow, queued, or unavailable.")
    if person is None:
        st.info("Upload a photo first.")
    else:
        ok = st.checkbox("I understand my photo will be sent to a third-party Hugging Face Space (yisol/IDM-VTON).")
        steps = st.slider("Denoising steps (more = better, slower)", 20, 40, 30)
        if st.button("✨ Generate photo-real try-on", disabled=not ok):
            try:
                from gradio_client import Client, handle_file
                garment, _ = build_garment(spec)
                with tempfile.TemporaryDirectory() as td:
                    pp, gp = os.path.join(td, "p.png"), os.path.join(td, "g.png")
                    person.save(pp)
                    core.flatten(garment).save(gp)
                    try:
                        client = Client("yisol/IDM-VTON", token=HF_TOKEN) if HF_TOKEN else Client("yisol/IDM-VTON")
                    except TypeError:
                        client = Client("yisol/IDM-VTON", hf_token=HF_TOKEN)
                    with st.spinner("Waiting for the GPU Space (30-120 s)..."):
                        res = client.predict(**{"dict": {"background": handle_file(pp), "layers": [], "composite": None},
                                                "garm_img": handle_file(gp), "garment_des": describe(spec),
                                                "is_checked": True, "is_checked_crop": False,
                                                "denoise_steps": steps, "seed": 42}, api_name="/tryon")
                    out = Image.open(res[0] if isinstance(res, (list, tuple)) else res).convert("RGB")
                c1, c2 = st.columns(2)
                c1.image(person, caption="You", use_container_width=True)
                c2.image(out, caption=describe(spec), use_container_width=True)
            except Exception as e:
                st.error("The community GPU Space did not answer (it may be busy, sleeping or out of free quota). "
                         "Try again later. Details: " + str(e)[:200])

# =============================================================== TAB 5
with tab5:
    st.markdown("""
### How it works
* **Clothes catalogue** - drawn in code (no copyrighted photos): tees, hoodie, blazer, dress, kurta ... in 20 colours and 4 patterns.
* **Fitting** - finds your face, sizes the garment from your face width, keeps your real face/hair on top and matches the lighting.
* **Does it suit me?** - reads your skin tone and undertone, groups you into a colour family (Spring / Summer / Autumn / Winter) and scores each colour on *palette match, contrast with your skin, and undertone harmony*.
* **GAN designer** - a **conditional GAN** (a generator and a discriminator trained against each other on fashion images) invents new garment shapes; the app tints and fits them.
* **Photo-real mode** - optional diffusion-model try-on running on a community GPU.

### Limits (be aware)
* Best with a front-facing photo, even light, plain top. Original clothes can peek out around the edges.
* GAN garments are 28×28 base images - concept sketches, not product photos.
* Scores are heuristics; lighting changes how skin tone is read.
""")
