import cv2
import torch
import numpy as np
import open_clip
from PIL import Image
from ultralytics import YOLO

class MaterialClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.autocast_device = "cuda" if self.device == "cuda" else "cpu"
        
        self.yolo = YOLO("./models/yolo26n.pt").to(self.device)
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.clip_model = self.clip_model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.labels = ["glass", "plastic", "metal", "paper", "wood", "fabric", "ceramic", "leather", "styrofoam"]
        self._init_text_features()

    def _init_text_features(self):
        prompts = [f"a photo of {m} material" for m in self.labels]
        tokens = self.tokenizer(prompts).to(self.device)
        
        with torch.inference_mode(), torch.amp.autocast(device_type=self.autocast_device):
            features = self.clip_model.encode_text(tokens)
            self.text_features = features / features.norm(dim=-1, keepdim=True)

    def predict_material(self, crop_img_bgr: np.ndarray) -> tuple:
        img_rgb = cv2.cvtColor(crop_img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        image_input = self.preprocess(img_pil).unsqueeze(0).to(self.device)

        with torch.inference_mode(), torch.amp.autocast(device_type=self.autocast_device):
            img_feat = self.clip_model.encode_image(image_input)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * img_feat.to(self.text_features.dtype) @ self.text_features.T).softmax(dim=-1)
            prob, index = similarity[0].max(dim=0)
            
        return self.labels[index.item()], float(prob.item())

    def get_padded_crop(self, img: np.ndarray, box) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        pw, ph = (x2 - x1) * 0.1, (y2 - y1) * 0.1
        x1_p, y1_p = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        x2_p, y2_p = min(w, int(x2 + pw)), min(h, int(y2 + ph))
        
        return img[y1_p:y2_p, x1_p:x2_p]