import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from transformers import (
#     OwlViTProcessor, OwlViTForObjectDetection
# )
from transformers import Owlv2Processor, Owlv2ForObjectDetection

@dataclass
class PilePosition:
    x: float
    y: float
    confidence: float
    label: str
    distance: float = 0.0
    box: Optional[List[float]] = None  # Added bounding box coordinates


class PileDetector:
    """Detect material piles using OWL-ViT with visualization support"""
    
    def __init__(self):
        print("🔍 Loading pile detection model...")
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        
        self.search_queries = [
            "pile of dirt", "pile of sand", "pile of gravel", 
            "construction material pile", "dirt pile", "sand pile",
            "material stockpile", "excavation pile", 
        ]
    
    def detect_piles(self, image_path: str) -> List[PilePosition]:
        """
        Detect piles in image and return positions with optional visualization
        
        Args:
            image_path: Path to input image
            visualize: Whether to display visualization
            save_visualization: Whether to save visualization to file
            output_path: Path to save visualization (if save_visualization=True)
        """
        try:
            image = Image.open(image_path)
            print(f"📸 Processing image: {image_path} (Size: {image.size})")
            
            inputs = self.processor(text=self.search_queries, images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            
            target_sizes = torch.Tensor([image.size[::-1]])
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.2
            )
            
            pile_positions = []
            if len(results[0]["boxes"]) > 0:
                print(f"🎯 Found {len(results[0]['boxes'])} potential detections")
                
                for i, (box, score, label) in enumerate(zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"])):
                    if score > 0.2:  # Confidence threshold
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
                            distance=distance,
                            box=[float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                        ))
                        
                        print(f"  Detection {i+1}: {self.search_queries[label]} "
                              f"(confidence: {score:.3f}, center: {x_center:.1f}, {y_center:.1f})")
            else:
                print("❌ No piles detected")
            
            # Sort by distance from center
            pile_positions = sorted(pile_positions, key=lambda p: p.distance)
            
            # self._visualize_detections(image, pile_positions)
            
            return pile_positions
            
        except Exception as e:
            print(f"❌ Error detecting piles in {image_path}: {e}")
            return []
    
    def _visualize_detections(self, image: Image.Image, detections: List[PilePosition], 
                             show: bool = False, save: bool = False, output_path: str = "pile_detection_results.png"):
        """Visualize detections with bounding boxes and labels"""
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title("Original Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Image with detections
        ax2.imshow(image)
        ax2.set_title(f"Detected Piles ({len(detections)} found)", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Colors for different detection types
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, detection in enumerate(detections):
            if detection.box:
                x1, y1, x2, y2 = detection.box
                color = colors[i % len(colors)]
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=3, edgecolor=color, facecolor='none'
                )
                ax2.add_patch(rect)
                
                # Add label with confidence
                label_text = f"{detection.label}\n{detection.confidence:.2f}"
                ax2.text(x1, y1-5, label_text, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                        fontsize=8, color='white', fontweight='bold')
                
                # Add center point
                ax2.plot(detection.x, detection.y, 'o', color=color, markersize=8)
                
                # Add distance annotation
                ax2.text(detection.x+10, detection.y+10, f"d={detection.distance:.0f}px", 
                        fontsize=8, color=color, fontweight='bold')
        
        # Add image center point for reference
        img_center_x, img_center_y = image.size[0]/2, image.size[1]/2
        ax2.plot(img_center_x, img_center_y, 'x', color='black', markersize=12, markeredgewidth=3)
        ax2.text(img_center_x+10, img_center_y-10, 'Image Center', 
                fontsize=10, color='black', fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = "pile_detection_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to: {output_path}")
        
        # Show visualization
        plt.show()
        
    def visualize_with_pil(self, image_path: str, detections: List[PilePosition], 
                          output_path: str = None) -> Image.Image:
        """
        Alternative visualization using PIL (lighter weight)
        """
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, detection in enumerate(detections):
            if detection.box:
                x1, y1, x2, y2 = detection.box
                color = colors[i % len(colors)]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw center point
                center_size = 5
                draw.ellipse([detection.x-center_size, detection.y-center_size, 
                            detection.x+center_size, detection.y+center_size], 
                           fill=color, outline='white', width=2)
                
                # Add label
                label_text = f"{detection.label[:15]}... {detection.confidence:.2f}"
                draw.text((x1, y1-25), label_text, fill=color, font=font)
        
        # Draw image center
        img_center_x, img_center_y = image.size[0]/2, image.size[1]/2
        draw.line([img_center_x-10, img_center_y, img_center_x+10, img_center_y], fill='black', width=3)
        draw.line([img_center_x, img_center_y-10, img_center_x, img_center_y+10], fill='black', width=3)
        
        if output_path:
            image.save(output_path)
            print(f"💾 PIL visualization saved to: {output_path}")
        
        return image
    
    def print_detection_summary(self, detections: List[PilePosition]):
        """Print a detailed summary of detections"""
        if not detections:
            print("❌ No detections found")
            return
        
        print(f"\n📊 Detection Summary ({len(detections)} piles found):")
        print("=" * 60)
        
        for i, detection in enumerate(detections, 1):
            print(f"Pile {i}:")
            print(f"  Label: {detection.label}")
            print(f"  Confidence: {detection.confidence:.3f}")
            print(f"  Center: ({detection.x:.1f}, {detection.y:.1f})")
            print(f"  Distance from center: {detection.distance:.1f}px")
            if detection.box:
                print(f"  Bounding box: ({detection.box[0]:.1f}, {detection.box[1]:.1f}, "
                      f"{detection.box[2]:.1f}, {detection.box[3]:.1f})")
            print()
