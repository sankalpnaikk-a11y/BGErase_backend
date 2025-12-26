import io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from rembg import remove, new_session
from PIL import Image

app = FastAPI(title="Background Remover API (HQ Portrait)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded session (CRITICAL)
SESSION = None

def get_session():
    global SESSION
    if SESSION is None:
        try:
            SESSION = new_session("u2net")
        except Exception as e:
            print("Model load failed:", e)
            raise
    return SESSION


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

        session = get_session()

        # ❌ alpha_matting disabled for memory safety
        cutout = remove(
            data=input_image,
            session=session,
            alpha_matting=False,
        )

        if not isinstance(cutout, Image.Image):
            cutout = Image.open(io.BytesIO(cutout)).convert("RGBA")

        alpha = cutout.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            cutout = cutout.crop(bbox)

        if bg_color:
            color = bg_color.lstrip("#")
            if len(color) == 3:
                color = "".join([c * 2 for c in color])
            if len(color) != 6:
                return JSONResponse(status_code=400, content={"error": "Invalid bg_color"})

            r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
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

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
