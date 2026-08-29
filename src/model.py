import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

class ModelNotLoadedError(Exception):
    pass

class VLMService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = None
        self.processor_name = None
        self.device = None
        self.is_ready = False

    def load(self, model_name: str, processor_name: str, device: str = "cuda"):
        if self.is_ready:
            return

        self.model_name = model_name
        self.processor_name = processor_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.processor_name, use_fast=True)
        self.model.eval()
        self.is_ready = True

    def _inference(self, image: Image.Image, label_map: dict[str, str]) -> dict[str, float]:
        if not self.is_ready:
            raise ModelNotLoadedError("Модель не загружена")

        clean_labels = list(label_map.keys())
        prompts = list(label_map.values())
        
        with torch.no_grad():
            inputs = self.processor(
                text=prompts, 
                images=image, 
                padding="max_length", 
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            if "siglip" in self.model_name.lower():
                probs = torch.sigmoid(outputs.logits_per_image)[0]
            else:
                probs = outputs.logits_per_image.softmax(dim=1)[0]

        return dict(zip(clean_labels, probs.tolist()))
    
    def predict(self, image: Image.Image, label_map: dict[str, str]):
        class_probs = self._inference(image=image, label_map=label_map)
        predicted_class = max(class_probs, key=class_probs.get)
        return predicted_class, class_probs
    
    
vlm_service = VLMService()