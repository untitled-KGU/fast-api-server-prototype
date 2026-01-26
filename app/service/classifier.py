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
        
        self.yolo = YOLO("yolov8x-oiv7.pt").to(self.device)
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.clip_model = self.clip_model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.categories = [
            "종이류", "종이팩", "금속캔", "고철", "유리병류", 
            "플라스틱 용기류", "비닐류", "의류 및 원단류", "폐가전제품", 
            "대형 폐기물", "음식물 쓰레기", "불연성 종량제", "종량제봉투", 
            "전용함", "전문시설", "주의", "재질별분리"
        ]
        
        self.prompt_templates = [
            "a photo of {label}",
            "the texture of {label}",
            "close-up of {label} material",
            "background of {label}",
            "an object made of {label}"
        ]

        self.raw_labels = [
            "paper and cardboard", "milk carton pack", "metal cans", 
            "scrap metal", "glass bottles", "plastic container", 
            "plastic vinyl", "clothes fabric", "electronic devices",
            "large furniture", "food waste", "ceramic porcelain", 
            "general garbage", "batteries", "medical waste", 
            "hazardous materials", "white styrofoam"
        ]
        
        self._init_features()

    def _init_features(self):
        final_features = []
        with torch.inference_mode(), torch.amp.autocast(device_type=self.autocast_device):
            for label in self.raw_labels:
                prompts = [t.format(label=label) for t in self.prompt_templates]
                tokens = self.tokenizer(prompts).to(self.device)
                embeddings = self.clip_model.encode_text(tokens)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                mean_embedding = embeddings.mean(dim=0)
                mean_embedding /= mean_embedding.norm(dim=-1)
                final_features.append(mean_embedding)
            
            self.text_features = torch.stack(final_features)

    def get_padded_crop(self, img: np.ndarray, box) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pw, ph = (x2 - x1) * 0.1, (y2 - y1) * 0.1
        x1_p, y1_p = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        x2_p, y2_p = min(w, int(x2 + pw)), min(h, int(y2 + ph))
        return img[y1_p:y2_p, x1_p:x2_p]

    def predict_material(self, crop_img_bgr: np.ndarray, obj_name: str) -> tuple:
        img_rgb = cv2.cvtColor(crop_img_bgr, cv2.COLOR_BGR2RGB)
        img_input = self.preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        obj_token = self.tokenizer([f"a photo of {obj_name}"]).to(self.device)

        with torch.inference_mode(), torch.amp.autocast(device_type=self.autocast_device):
            img_feat = self.clip_model.encode_image(img_input)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
            obj_feat = self.clip_model.encode_text(obj_token)
            obj_feat /= obj_feat.norm(dim=-1, keepdim=True)
            
            combined_feat = (img_feat * 0.85) + (obj_feat * 0.15)
            combined_feat /= combined_feat.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * combined_feat @ self.text_features.T).softmax(dim=-1)
            prob, index = similarity[0].max(dim=0)

        return self.categories[index.item()], float(prob.item())