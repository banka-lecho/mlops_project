import torch
import numpy as np
from transformers import AutoProcessor, AutoModel

class VLMModel:
    def __init__(self, model_name: str, processor_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.processor_name = processor_name
        self.device = device
        
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(self.processor_name, use_fast=True)
        
        self.model.eval()
        
    def _inference(self, image: np.ndarray, label_map: list[str]) -> dict[str, float]:
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
    
    def _postprocess(self):
        pass
    
    def predict(self, image: np.ndarray, label_map: dict[str, str]):
        class_probs = self._inference(image=image, label_map=label_map)
        predicted_class = max(class_probs, key=class_probs.get)
        return predicted_class