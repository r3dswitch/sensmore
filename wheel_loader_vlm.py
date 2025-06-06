import os
import json
import torch
from transformers import (
    pipeline
)
from typing import List, Dict
from dataclasses import dataclass
from pile_detector import PileDetector
from vqa_generator import VQAGenerator
from action_command_generator import ActionCommandGenerator
from action_visualizer import ActionVisualizer

@dataclass
class PilePosition:
    x: float
    y: float
    confidence: float
    label: str
    distance: float = 0.0

class WheelLoaderVLM:
    """Main Vision-Language-Action Model for Wheel Loader"""
    
    def __init__(self):
        print("🤖 Initializing Wheel Loader VLM...")
        
        self.pile_detector = PileDetector()
        self.vqa_generator = VQAGenerator()
        self.command_generator = ActionCommandGenerator()
        self.visualizer = ActionVisualizer()
        
        # Initialize LLM (using a lightweight model for demo)
        print("🧠 Loading language model...")
        self.llm_pipeline = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-small",  # Lightweight alternative
            tokenizer="microsoft/DialoGPT-small",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def process_command(self, image_path: str, text_command: str, output_path: str) -> Dict:
        """Main processing pipeline"""
        print(f"🔄 Processing command: '{text_command}' for image: {image_path}")
        
        # Step 1: Detect piles
        pile_positions = self.pile_detector.detect_piles(image_path)
        print(f"   Found {len(pile_positions)} piles")
        
        # Step 2: Create context for LLM
        context = self._create_context(image_path, pile_positions, text_command)
        
        # Step 3: Generate LLM response
        llm_response = self._generate_response(context)
        print(f"   LLM Response: {llm_response}")
        
        # Step 4: Convert to commands
        action_commands = self.command_generator.text_to_commands(llm_response, pile_positions)
        print(f"   Action Commands: {action_commands}")
        
        self.visualizer.visualize_actions(image_path, pile_positions, action_commands, output_path)

        return {
            "image_path": image_path,
            "input_command": text_command,
            "detected_piles": [
                {
                    "position": (p.x, p.y),
                    "confidence": p.confidence,
                    "label": p.label,
                    "distance": p.distance
                } for p in pile_positions
            ],
            "llm_response": llm_response,
            "action_commands": action_commands
        }
    
    def _create_context(self, image_path: str, pile_positions: List[PilePosition], command: str) -> str:
        """Create context string for LLM"""
        pile_info = ""
        if pile_positions:
            pile_info = f"Detected {len(pile_positions)} material piles:\n"
            for i, pile in enumerate(pile_positions[:3]):  # Top 3 piles
                pile_info += f"- Pile {i+1}: {pile.label} at ({pile.x:.0f}, {pile.y:.0f}), confidence: {pile.confidence:.2f}\n"
        else:
            pile_info = "No material piles detected in the image.\n"
        
        context = f"""
                    Construction Site Analysis:
                    {pile_info}
                    Operator Command: {command}

                    As a wheel loader operator assistant, provide specific instructions for the loader to complete this task.
                    Focus on safe and efficient operation. Include position coordinates when possible.

                    Response:"""
        
        return context
    
    def _generate_response(self, context: str) -> str:
        """Generate response using LLM"""
        try:
            # return self._rule_based_response(context)
            return self._llm_based_response(context)
            
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return "Drive to the nearest pile and begin digging operation."
    
    def _rule_based_response(self, context: str) -> str:
        """Simple rule-based response for demonstration"""
        if "no material piles detected" in context.lower():
            return "No piles detected. Search the construction site for material stockpiles."
        
        # Extract pile information
        lines = context.split('\n')
        pile_lines = [line for line in lines if 'Pile' in line and 'at (' in line]
        
        if pile_lines:
            # Parse first pile position
            import re
            match = re.search(r'at \((\d+), (\d+)\)', pile_lines[0])
            if match:
                x, y = match.groups()
                return f"Drive to position ({x}, {y}) and position the bucket for optimal digging angle. Begin excavation once positioned."
        
        return "Approach the nearest visible pile and begin digging operation."
    
    def _llm_based_response(self, context: str) -> str:
        """Generate response using SmolLM with specialized construction prompting"""
        if not self.llm_pipeline:
            return self._rule_based_response(context)
            
        try:
            # Extract key information from context
            pile_info = self._extract_pile_info(context)
            command_info = self._extract_command_info(context)
            
            # Create specialized prompt for construction operations
            prompt = f"""<|im_start|>system You are an expert wheel loader operator assistant. Analyze construction site data and provide precise, actionable instructions.

                Key responsibilities:
                - Prioritize operator safety and equipment protection
                - Give specific coordinates and positioning instructions  
                - Use professional construction terminology
                - Provide step-by-step operational guidance
                - Consider optimal digging angles and approach paths

                Always respond with concrete actions, not questions or generic advice.<|im_end|>
                <|im_start|>user
                CONSTRUCTION SITE ANALYSIS:
                {pile_info}

                OPERATOR REQUEST: {command_info}

                Provide specific operational instructions including:
                1. Movement/positioning commands
                2. Equipment positioning details  
                3. Execution sequence
                4. Safety considerations

                Response:<|im_end|>
                <|im_start|>assistant
                """
            
            # Generate response with optimized parameters for construction domain
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=120,
                temperature=0.3,  # Lower temperature for more focused responses
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            if response and len(response) > 0:
                generated_text = response[0]['generated_text'].strip()
                
                # Clean up response
                generated_text = self._clean_response(generated_text)
                
                # Validate response quality
                if self._is_valid_construction_response(generated_text):
                    return generated_text
                else:
                    print("⚠️ SmolLM response quality check failed, using rule-based fallback")
                    return self._rule_based_response(context)
            else:
                return self._rule_based_response(context)
                
        except Exception as e:
            print(f"SmolLM generation error: {e}")
            return self._rule_based_response(context)

    def create_training_dataset(self, image_paths: List[str]) -> List[Dict]:
        """Create training dataset from images"""
        print("📚 Creating training dataset...")
        dataset = []
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                continue
                
            # Detect piles
            pile_positions = self.pile_detector.detect_piles(image_path)
            
            # Generate VQA pairs
            vqa_pairs = self.vqa_generator.generate_vqa_pairs(image_path, pile_positions)
            
            dataset.extend(vqa_pairs)
        
        print(f"✅ Generated {len(dataset)} training samples")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Save dataset to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"💾 Dataset saved to {filepath}")
    
    def demo_interactive(self):
        """Interactive demo mode"""
        print("\n🎮 Interactive Wheel Loader VLM Demo")
        print("Commands: 'fill bucket', 'dig here', 'find pile', 'quit'")
        
        while True:
            image_path = input("\nEnter image path (or 'quit'): ").strip()
            if image_path.lower() == 'quit':
                break
                
            if not os.path.exists(image_path):
                print("❌ Image not found!")
                continue
            
            command = input("Enter command: ").strip()
            if command.lower() == 'quit':
                break
            
            result = self.process_command(image_path, command, "res.png")
            
            print("\n📊 Results:")
            print(f"Detected Piles: {len(result['detected_piles'])}")
            for pile in result['detected_piles']:
                print(f"  - {pile['label']} at {pile['position']} (conf: {pile['confidence']:.2f})")
            
            print(f"\nLLM Response: {result['llm_response']}")
            print(f"Action Commands: {result['action_commands']}")

