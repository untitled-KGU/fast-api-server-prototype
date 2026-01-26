import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from app.service.classifier import MaterialClassifier
from app.utils.storage import ImageStorage
from app.model.request import ExtractRequest

app = FastAPI()
classifier = MaterialClassifier()
image_storage = ImageStorage()

@app.post("/extracts")
async def extract_and_crop(data: ExtractRequest):
    request_id = data.request_id
    img_pil = image_storage.get_origin_image(request_id)        
    if img_pil is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    results = classifier.yolo.predict(source=img, conf=0.4, verbose=False)
    detected_items = []

    for result in results:
        for box in result.boxes:
            crop = classifier.get_padded_crop(img, box)
            if crop.size == 0: continue

            obj_name = classifier.yolo.names[int(box.cls[0])]
            material, prob = classifier.predict_material(crop, obj_name)
            filename = image_storage.save_crop(crop, request_id)

            detected_items.append({
                "label": obj_name,
                "detail": material,
                "confidence": {
                    "object": round(float(box.conf[0]), 3),
                    "detail": round(prob, 3)
                },
                "filename": filename
            })

    return {
        "request_id": request_id,
        "count": len(detected_items),
        "detected_items": detected_items
    }