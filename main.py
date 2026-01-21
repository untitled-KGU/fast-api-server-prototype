import os
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, status
from app.service.classifier import MaterialClassifier
from app.utils.storage import ImageStorage
from app.model.request import ExtractRequest
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
classifier = MaterialClassifier()
image_storage = ImageStorage()


@app.post("/extracts")
async def extract_and_crop(
    data: ExtractRequest
):
    request_id = data.request_id
    img = image_storage.get_origin_image(request_id)        
    if img is None:
        target_path = os.path.join(image_storage.origin_dir, f"{request_id}.jpg")
        if not os.path.exists(target_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image file not found on server."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format or corrupted file."
            )
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = classifier.yolo.predict(source=img, conf=0.6, verbose=False)
    detected_items = []

    for result in results:
        for box in result.boxes:
            crop = classifier.get_padded_crop(img, box)
            if crop.size == 0: continue

            material, prob = classifier.predict_material(crop)
            filename = image_storage.save_crop(crop, request_id)

            detected_items.append({
                "label": classifier.yolo.names[int(box.cls[0])],
                "material": material,
                "confidence": {
                    "object": round(float(box.conf[0]), 3),
                    "material": round(prob, 3)
                },
                "filename": filename
            })

    return {
        "request_id": request_id,
        "count": len(detected_items),
        "detected_items": detected_items
    }