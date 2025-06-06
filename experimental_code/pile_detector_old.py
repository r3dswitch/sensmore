import torch
import numpy as np
from PIL import Image
from typing import List
from dataclasses import dataclass
from transformers import (
    OwlViTProcessor, OwlViTForObjectDetection
)

@dataclass
class PilePosition:
    x: float
    y: float
    confidence: float
    label: str
    distance: float = 0.0


class PileDetector:
    """Detect material piles using OWL-ViT"""
    
    def __init__(self):
        print("🔍 Loading pile detection model...")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        
        self.search_queries = [
            "pile of dirt", "pile of sand", "pile of gravel", 
            "construction material pile", "dirt pile", "sand pile",
            "material stockpile", "excavation pile", 
        ]
    
    def detect_piles(self, image_path: str) -> List[PilePosition]:
        """Detect piles in image and return positions"""
        try:
            image = Image.open(image_path)
            
            inputs = self.processor(text=self.search_queries, images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            
            target_sizes = torch.Tensor([image.size[::-1]])
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.05
            )
            
            pile_positions = []
            if len(results[0]["boxes"]) > 0:
                for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
                    if score > 0.05:  # Confidence threshold
                        x_center = float((box[0] + box[2]) / 2)
                        y_center = float((box[1] + box[3]) / 2)
                        
                        # Calculate distance from center (simple heuristic)
                        img_center_x, img_center_y = image.size[0]/2, image.size[1]/2
                        distance = np.sqrt((x_center - img_center_x)**2 + (y_center - img_center_y)**2)
                        
                        pile_positions.append(PilePosition(
                            x=x_center,
                            y=y_center,
                            confidence=float(score),
                            label=self.search_queries[label],
                            distance=distance
                        ))
            return sorted(pile_positions, key=lambda p: p.distance)  # Sort by distance
            
        except Exception as e:
            print(f"Error detecting piles in {image_path}: {e}")
            return []
