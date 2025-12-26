import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from rembg import remove, new_session
from PIL import Image

# ✅ CREATE APP ONLY ONCE
app = FastAPI(title="Background Remover API (HQ Portrait)")

# Try portrait-optimized model, fall back to generic U2Net
try:
    SESSION = new_session("u2net_human_seg")
except Exception:
    SESSION = new_session("u2net")

# CORS
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
            color = bg_color.lstrip("#")
            r, g, b = int(color[0:2],16), int(color[2:4],16), int(color[4:6],16)
            bg = Image.new("RGBA", cutout.size, (r, g, b, 255))
            bg.paste(cutout, mask=cutout.split()[3])
            output = bg
        else:
            output = cutout

        if width and height:
            output = output.resize((width, height), Image.LANCZOS)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
