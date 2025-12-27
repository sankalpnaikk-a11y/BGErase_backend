import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from rembg import remove, new_session
from PIL import Image
import gc  # ✅ ADDED (memory cleanup)

app = FastAPI(title="Background Remover API (HQ Portrait)")

# =========================
# 🔹 ADDED: lazy-loaded session
# =========================
SESSION = None


# =========================
# 🔹 ADDED: startup hook
# =========================
@app.on_event("startup")
def load_model():
    global SESSION
    try:
        print("Loading u2net_human_seg model...")
        SESSION = new_session("u2net_human_seg")
        print("Loaded portrait model")
    except Exception as e:
        print("Falling back to u2net:", e)
        SESSION = new_session("u2net")


# CORS for frontend (Render / Vercel / local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Background Remover API running"}


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    bg_color: Optional[str] = Form(None),
):
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGBA")

        cutout = remove(
            data=input_image,
            session=SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=245,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=5,
        )

        if not isinstance(cutout, Image.Image):
            cutout = Image.open(io.BytesIO(cutout)).convert("RGBA")

        alpha = cutout.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            cutout = cutout.crop(bbox)

        if bg_color:
            color = bg_color.strip()
            if color.startswith("#"):
                color = color[1:]
            if len(color) == 3:
                color = "".join([c * 2 for c in color])
            if len(color) != 6:
                return JSONResponse(
                    status_code=400,
                    content={"error": "bg_color must be a valid hex color like #ffffff"},
                )

            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)

            background = Image.new("RGBA", cutout.size, (r, g, b, 255))
            background.paste(cutout, mask=cutout.split()[3])
            output_image = background
        else:
            output_image = cutout

        if width and height:
            output_image = output_image.resize((width, height), Image.LANCZOS)

        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        buf.seek(0)

        # 🔹 ADDED: explicit memory cleanup
        del input_image, cutout
        gc.collect()

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="output.png"'},
        )

    except Exception as e:
        print("ERROR in /process-image:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
