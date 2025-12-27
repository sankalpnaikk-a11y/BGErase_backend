import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from rembg import remove, new_session
from PIL import Image
import asyncio

app = FastAPI(title="BGErase – Background Remover API (U2Net refined)")

# -----------------------------
# MODEL: U2Net (lazy load on startup)
# -----------------------------
SESSION = None
MAX_MODEL_SIDE = 1800  # a bit larger than before for more detail

async def _load_model():
    """Load the heavy model in a thread so startup is non-blocking."""
    global SESSION
    SESSION = await asyncio.to_thread(new_session, "u2net")

@app.on_event("startup")
async def startup_event():
    # start loading in background (don't await) so the server binds the port quickly
    asyncio.create_task(_load_model())


# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "ok", "model": "u2net"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": SESSION is not None}


# -----------------------------
# Small helper to tighten alpha (cleaner edges)
# -----------------------------
def tighten_alpha(alpha_img: Image.Image) -> Image.Image:
    """
    Increase alpha contrast so edges are cleaner:
    - very small alpha -> 0 (fully transparent)
    - very large alpha -> 255 (fully opaque)
    - middle range stretched.
    """
    lut = []
    low = 10      # below this -> 0
    high = 235    # above this -> 255
    rng = high - low if high > low else 1

    for i in range(256):
        if i <= low:
            lut.append(0)
        elif i >= high:
            lut.append(255)
        else:
            # scale [low, high] -> [0, 255]
            val = int((i - low) * 255 / rng)
            lut.append(val)

    return alpha_img.point(lut)


# -----------------------------
# MAIN ENDPOINT
# -----------------------------
@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    bg_color: Optional[str] = Form(None),      # "" or None -> transparent
    format: Optional[str] = Form("png"),       # "png" | "jpeg" | "webp"
    quality: Optional[int] = Form(90),         # for jpeg/webp
):
    try:
        if SESSION is None:
            return JSONResponse(status_code=503, content={"error": "model_loading"})
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGBA")
        orig_w, orig_h = input_image.size

        # 1) Downscale big images for speed
        max_side = max(orig_w, orig_h)
        scale = 1.0
        work_image = input_image

        if max_side > MAX_MODEL_SIDE:
            scale = MAX_MODEL_SIDE / float(max_side)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            work_image = input_image.resize((new_w, new_h), Image.LANCZOS)

        # 2) Background removal with alpha matting
        cutout_small = remove(
            data=work_image,
            session=SESSION,
            alpha_matting=True,
            # slightly stricter FG, soft BG to keep hair but clean body edges
            alpha_matting_foreground_threshold=245,
            alpha_matting_background_threshold=15,
            alpha_matting_erode_size=2,
        )

        # scale back up to original size if we downscaled
        if scale != 1.0:
            cutout = cutout_small.resize((orig_w, orig_h), Image.LANCZOS)
        else:
            cutout = cutout_small

        # 2b) Tighten alpha so edges are crisp (important on solid backgrounds)
        r, g, b, a = cutout.split()
        a = tighten_alpha(a)
        cutout = Image.merge("RGBA", (r, g, b, a))

        # 3) Background handling
        if bg_color:
            color = bg_color.strip()
            if color.startswith("#"):
                color = color[1:]
            if len(color) == 3:
                color = "".join([c * 2 for c in color])
            if len(color) != 6:
                return JSONResponse(
                    status_code=400,
                    content={"error": "bg_color must be a valid hex like #ffffff"},
                )

            r_bg = int(color[0:2], 16)
            g_bg = int(color[2:4], 16)
            b_bg = int(color[4:6], 16)

            background = Image.new("RGBA", cutout.size, (r_bg, g_bg, b_bg, 255))
            # paste using refined alpha mask
            background.paste(cutout, mask=a)
            output_image = background
        else:
            output_image = cutout  # transparent

        # 4) Optional resize (passport, presets, etc.)
        if width and height:
            output_image = output_image.resize((width, height), Image.LANCZOS)

        # 5) Encode & send
        buf = io.BytesIO()
        fmt = (format or "png").lower()

        if fmt == "jpeg":
            output_image = output_image.convert("RGB")
            output_image.save(buf, format="JPEG", quality=quality or 90, optimize=True)
            media_type = "image/jpeg"
            filename = "output.jpg"
        elif fmt == "webp":
            output_image.save(buf, format="WEBP", quality=quality or 90, method=6)
            media_type = "image/webp"
            filename = "output.webp"
        else:
            output_image.save(buf, format="PNG", optimize=True)
            media_type = "image/png"
            filename = "output.png"

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
