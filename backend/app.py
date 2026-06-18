import json
import io

import torch
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_def import PlantDiseaseCNN

app = FastAPI(title="Plant Disease Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("class_names.json") as f:
    class_names = json.load(f)

with open("model_config.json") as f:
    config = json.load(f)

model = PlantDiseaseCNN(num_classes=config["num_classes"]).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": "PlantDiseaseCNN",
        "classes": len(class_names),
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPG / PNG / WEBP images allowed")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        pred_idx    = probs.argmax(dim=1).item()
        confidence  = probs[0][pred_idx].item()

    top3_probs, top3_idxs = torch.topk(probs, 3, dim=1)
    top3 = [
        {
            "class":      class_names[top3_idxs[0][i].item()],
            "confidence": round(top3_probs[0][i].item() * 100, 2)
        }
        for i in range(3)
    ]

    predicted_class = class_names[pred_idx]
    # split "Plant___Disease" → plant name + disease label
    parts      = predicted_class.replace("___", "__").split("__")
    plant      = parts[0].replace("_", " ") if len(parts) > 0 else predicted_class
    disease    = parts[1].replace("_", " ") if len(parts) > 1 else ""
    is_healthy = "healthy" in predicted_class.lower()

    return {
        "predicted_class": predicted_class,
        "plant":           plant,
        "disease":         disease,
        "is_healthy":      is_healthy,
        "confidence":      round(confidence * 100, 2),
        "top3":            top3
    }


@app.get("/classes")
def get_classes():
    return {"total": len(class_names), "classes": class_names}