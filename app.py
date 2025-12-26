import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from rembg import remove, new_session
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    return {"status": "working"}


app = FastAPI(title="Background Remover API (HQ Portrait)")

# Try portrait-optimized model, fall back to generic U2Net
try:
    SESSION = new_session("u2net_human_seg")
except Exception:
    # Safe fallback – always available
    SESSION = new_session("u2net")


# CORS for local dev
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
    bg_color: Optional[str] = Form(
        None
    ),  # hex like "#ffffff" or ""/None for transparent
):
    try:
        # Read upload into PIL image
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGBA")

        # High-quality background removal with alpha matting
        # High-quality background removal with alpha matting
        cutout = remove(
            data=input_image,
            session=SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=245,  # stricter on subject
            alpha_matting_background_threshold=20,   # confident about background
            alpha_matting_erode_size=5,              # keeps fine edges
        )

        # At this point `cutout` is a PIL RGBA image
        if not isinstance(cutout, Image.Image):
            cutout = Image.open(io.BytesIO(cutout)).convert("RGBA")

        # Crop to bounding box of non-transparent pixels BEFORE adding background or resizing
        alpha = cutout.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            cutout = cutout.crop(bbox)

        # Background handling
        # If bg_color is empty or None -> keep transparent
        if bg_color:
            color = bg_color.strip()
            if color.startswith("#"):
                color = color[1:]
            if len(color) == 3:
                # fff -> ffffff
                color = "".join([c * 2 for c in color])
            if len(color) != 6:
                return JSONResponse(
                    status_code=400,
                    content={"error": "bg_color must be a valid hex color like #ffffff"},
                )

            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)

            # Solid color background with our cutout pasted using alpha mask
            background = Image.new("RGBA", cutout.size, (r, g, b, 255))
            # use alpha channel as mask
            background.paste(cutout, mask=cutout.split()[3])
            output_image = background
        else:
            # Keep transparent background
            output_image = cutout

        # Optional resize if both width & height provided
        if width and height:
            output_image = output_image.resize((width, height), Image.LANCZOS)

        # Optional resize if both width & height provided
        if width and height:
            output_image = output_image.resize((width, height), Image.LANCZOS)

        # Return PNG bytes
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="output.png"'},
        )

    except Exception as e:
        # Log error to server console and send JSON to client
        print("ERROR in /process-image:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
