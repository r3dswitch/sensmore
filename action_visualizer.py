import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re

@dataclass
class PilePosition:
    x: float
    y: float
    confidence: float
    label: str
    distance: float = 0.0
    box: Optional[List[float]] = None  # Added bounding box coordinates


class ActionVisualizer:
    """Handle all visualization tasks for pile detection and action commands"""
    
    def __init__(self):
        pass
    
    def visualize_action_command(self, image: Image.Image, detections: List[PilePosition], 
                                action_commands: List[str] = None, output_path: str = "action_visualization.png"):
        """
        Visualize action command with bounding boxes, action sequence, and movement lines
        
        Args:
            image: PIL Image to visualize on
            detections: List of detected pile positions
            action_commands: List of action command strings (e.g., ["DRIVE_TO(316.0, 79.0)", "DIG_AT_POSITION(316.0, 79.0)"])
            output_path: Path to save the visualization
        """
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        ax.imshow(image)
        ax.set_title("Action Command Visualization", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Colors for different elements
        detection_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        action_color = 'yellow'
        path_color = 'lime'
        
        # Draw detections
        for i, detection in enumerate(detections):
            if detection.box:
                x1, y1, x2, y2 = detection.box
                color = detection_colors[i % len(detection_colors)]
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
                )
                ax.add_patch(rect)
                
                # Add detection label
                label_text = f"Pile {i+1}\n{detection.confidence:.2f}"
                ax.text(x1, y1-5, label_text, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                       fontsize=9, color='white', fontweight='bold')
                
                # Add center point
                ax.plot(detection.x, detection.y, 'o', color=color, markersize=6)
        
        # Parse and visualize action commands
        if action_commands:
            current_pos = None
            action_positions = []
            
            # Extract positions from action commands
            for i, command in enumerate(action_commands):
                # Parse DRIVE_TO and DIG_AT_POSITION commands
                drive_match = re.search(r'DRIVE_TO\((\d+\.?\d*),\s*(\d+\.?\d*)\)', command)
                dig_match = re.search(r'DIG_AT_POSITION\((\d+\.?\d*),\s*(\d+\.?\d*)\)', command)
                bucket_match = re.search(r'POSITION_BUCKET\((\d+)\)', command)
                
                if drive_match:
                    x, y = float(drive_match.group(1)), float(drive_match.group(2))
                    action_positions.append({
                        'type': 'DRIVE_TO',
                        'position': (x, y),
                        'command': command,
                        'step': i + 1
                    })
                    current_pos = (x, y)
                
                elif dig_match:
                    x, y = float(dig_match.group(1)), float(dig_match.group(2))
                    action_positions.append({
                        'type': 'DIG_AT_POSITION',
                        'position': (x, y),
                        'command': command,
                        'step': i + 1
                    })
                    current_pos = (x, y)
                
                elif bucket_match:
                    angle = int(bucket_match.group(1))
                    if current_pos:  # Use last known position
                        action_positions.append({
                            'type': 'POSITION_BUCKET',
                            'position': current_pos,
                            'angle': angle,
                            'command': command,
                            'step': i + 1
                        })
            
            # Draw action sequence
            start_pos = (image.size[0] // 2, image.size[1] - 50)  # Assume starting position at bottom center
            
            # Draw path from start to first action
            if action_positions:
                first_pos = action_positions[0]['position']
                ax.plot([start_pos[0], first_pos[0]], [start_pos[1], first_pos[1]], 
                       color=path_color, linewidth=4, linestyle='--', alpha=0.8, label='Movement Path')
                
                # Add arrow to show direction
                ax.annotate('', xy=first_pos, xytext=start_pos,
                           arrowprops=dict(arrowstyle='->', color=path_color, lw=3))
            
            # Draw starting position
            ax.plot(start_pos[0], start_pos[1], 's', color='black', markersize=12, 
                   markeredgecolor='white', markeredgewidth=2, label='Start Position')
            ax.text(start_pos[0], start_pos[1]-25, 'START', ha='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8),
                   fontsize=10, color='white', fontweight='bold')
            
            # Draw action positions and connections
            prev_pos = None
            for i, action in enumerate(action_positions):
                pos = action['position']
                
                # Draw connection line between consecutive actions
                if prev_pos and prev_pos != pos:
                    ax.plot([prev_pos[0], pos[0]], [prev_pos[1], pos[1]], 
                           color=path_color, linewidth=3, alpha=0.8)
                    # Add arrow
                    ax.annotate('', xy=pos, xytext=prev_pos,
                               arrowprops=dict(arrowstyle='->', color=path_color, lw=2))
                
                # Draw action marker
                if action['type'] == 'DRIVE_TO':
                    marker = '^'
                    color = 'blue'
                    size = 10
                elif action['type'] == 'DIG_AT_POSITION':
                    marker = 'D'
                    color = 'red'
                    size = 12
                elif action['type'] == 'POSITION_BUCKET':
                    marker = 'h'
                    color = 'orange'
                    size = 10
                
                ax.plot(pos[0], pos[1], marker, color=color, markersize=size, 
                       markeredgecolor='white', markeredgewidth=2)
                
                # Add action label
                if action['type'] == 'POSITION_BUCKET':
                    label_text = f"Step {action['step']}: {action['type']}\nAngle: {action['angle']}°"
                else:
                    label_text = f"Step {action['step']}: {action['type']}\n({pos[0]:.1f}, {pos[1]:.1f})"
                
                # Position label to avoid overlap
                offset_y = -40 if i % 2 == 0 else 20
                ax.text(pos[0], pos[1] + offset_y, label_text, ha='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8),
                       fontsize=9, color='white', fontweight='bold')
                
                prev_pos = pos
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=path_color, lw=3, linestyle='--', label='Movement Path'),
            plt.Line2D([0], [0], marker='s', color='black', linestyle='None', 
                      markersize=8, label='Start Position'),
            plt.Line2D([0], [0], marker='^', color='blue', linestyle='None', 
                      markersize=8, label='Drive To'),
            plt.Line2D([0], [0], marker='D', color='red', linestyle='None', 
                      markersize=8, label='Dig Position'),
            plt.Line2D([0], [0], marker='h', color='orange', linestyle='None', 
                      markersize=8, label='Bucket Position')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add command text box
        if action_commands:
            command_text = "Action Commands:\n" + "\n".join([f"{i+1}. {cmd}" for i, cmd in enumerate(action_commands)])
            ax.text(0.02, 0.98, command_text, transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
                   verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"🎯 Action visualization saved to: {output_path}")
        plt.close()
        
        return output_path
    
    def visualize_actions(self, image_path: str, detections: List[PilePosition], 
                         action_commands: List[str], output_path: str = "action_plan.png"):
        """
        Convenience method to visualize action commands on an image
        
        Args:
            image_path: Path to the image file
            detections: List of detected pile positions
            action_commands: List of action command strings
            output_path: Path to save the visualization
        """
        try:
            image = Image.open(image_path)
            return self.visualize_action_command(image, detections, action_commands, output_path)
        except Exception as e:
            print(f"❌ Error visualizing actions: {e}")
            return None
    
    def visualize_detections(self, image: Image.Image, detections: List[PilePosition], 
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