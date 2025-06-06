import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class PilePosition:
    """Import or redefine the PilePosition class"""
    x: float
    y: float
    confidence: float
    label: str
    distance: float = 0.0


class ActionVisualizer:
    """Handles visualization of pile detection results and action commands"""
    
    def __init__(self):
        self.colors = {
            'pile': (255, 0, 0),      # Red for piles
            'action': (0, 255, 0),    # Green for actions
            'path': (0, 0, 255),      # Blue for paths
            'text': (255, 255, 255),  # White for text
            'background': (0, 0, 0)   # Black for text background
        }
        
    def visualize_actions(self, 
                         image_path: str, 
                         pile_positions: List[PilePosition], 
                         action_commands: List[Dict[str, Any]], 
                         output_path: str) -> None:
        """
        Visualize detected piles and action commands on the image
        
        Args:
            image_path: Path to the input image
            pile_positions: List of detected pile positions
            action_commands: List of action command dictionaries
            output_path: Path to save the visualization
        """
        try:
            # Load image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                small_font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw detected piles
            self._draw_piles(draw, pile_positions, font)
            
            # Draw action commands
            self._draw_actions(draw, action_commands, font, small_font)
            
            # Draw legend
            self._draw_legend(draw, image.size, small_font)
            
            # Save the result
            image.save(output_path)
            print(f"✅ Visualization saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
    
    def _draw_piles(self, draw: ImageDraw.Draw, pile_positions: List[PilePosition], font: ImageFont) -> None:
        """Draw pile detection results"""
        for i, pile in enumerate(pile_positions):
            # Draw pile marker (circle)
            radius = 10
            bbox = [pile.x - radius, pile.y - radius, pile.x + radius, pile.y + radius]
            draw.ellipse(bbox, outline=self.colors['pile'], width=3)
            
            # Draw pile number
            draw.text((pile.x + radius + 5, pile.y - radius), 
                     f"P{i+1}", fill=self.colors['pile'], font=font)
            
            # Draw confidence score
            confidence_text = f"{pile.confidence:.2f}"
            draw.text((pile.x + radius + 5, pile.y + 5), 
                     confidence_text, fill=self.colors['pile'], font=font)
    
    def _draw_actions(self, draw: ImageDraw.Draw, action_commands: List[Dict[str, Any]], 
                     font: ImageFont, small_font: ImageFont) -> None:
        """Draw action commands"""
        for i, action in enumerate(action_commands):
            if 'position' in action:
                pos = action['position']
                x, y = pos.get('x', 0), pos.get('y', 0)
                
                # Draw action marker (square)
                size = 8
                bbox = [x - size, y - size, x + size, y + size]
                draw.rectangle(bbox, outline=self.colors['action'], width=2)
                
                # Draw action type
                action_type = action.get('type', 'unknown')
                draw.text((x + size + 5, y - size), 
                         f"A{i+1}: {action_type}", fill=self.colors['action'], font=small_font)
    
    def _draw_legend(self, draw: ImageDraw.Draw, image_size: tuple, font: ImageFont) -> None:
        """Draw a simple legend"""
        legend_x = 10
        legend_y = image_size[1] - 80
        
        # Background for legend
        legend_bg = [legend_x - 5, legend_y - 5, legend_x + 200, legend_y + 70]
        draw.rectangle(legend_bg, fill=self.colors['background'], outline=self.colors['text'])
        
        # Legend items
        draw.text((legend_x, legend_y), "Legend:", fill=self.colors['text'], font=font)
        draw.text((legend_x, legend_y + 20), "● Piles (Red circles)", fill=self.colors['pile'], font=font)
        draw.text((legend_x, legend_y + 40), "■ Actions (Green squares)", fill=self.colors['action'], font=font)
    
    def visualize_with_opencv(self, 
                             image_path: str, 
                             pile_positions: List[PilePosition], 
                             action_commands: List[Dict[str, Any]], 
                             output_path: str) -> None:
        """
        Alternative visualization using OpenCV (if you prefer OpenCV over PIL)
        """
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Draw piles
            for i, pile in enumerate(pile_positions):
                center = (int(pile.x), int(pile.y))
                cv2.circle(image, center, 10, self.colors['pile'][::-1], 2)  # BGR format
                cv2.putText(image, f"P{i+1}", (int(pile.x + 15), int(pile.y - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['pile'][::-1], 2)
                cv2.putText(image, f"{pile.confidence:.2f}", (int(pile.x + 15), int(pile.y + 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['pile'][::-1], 1)
            
            # Draw actions
            for i, action in enumerate(action_commands):
                if 'position' in action:
                    pos = action['position']
                    x, y = int(pos.get('x', 0)), int(pos.get('y', 0))
                    cv2.rectangle(image, (x-8, y-8), (x+8, y+8), self.colors['action'][::-1], 2)
                    action_type = action.get('type', 'unknown')
                    cv2.putText(image, f"A{i+1}: {action_type}", (x + 15, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['action'][::-1], 1)
            
            # Save result
            cv2.imwrite(output_path, image)
            print(f"✅ OpenCV visualization saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Error creating OpenCV visualization: {e}")


# Convenience function for easy import
def create_visualizer() -> ActionVisualizer:
    """Factory function to create a new ActionVisualizer instance"""
    return ActionVisualizer()