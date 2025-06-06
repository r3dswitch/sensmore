import random
from typing import List, Dict
from pile_detector import PilePosition

class VQAGenerator:
    """Generate Vision-Question-Answer pairs automatically"""
    
    def __init__(self):
        self.question_templates = {
            "location": [
                "Where is the nearest pile?",
                "Which pile should I target first?",
                "Where should the loader go next?",
                "What is the best pile to dig from?"
            ],
            "action": [
                "What action should the loader take?",
                "How should the loader approach the pile?",
                "What's the next step for filling the bucket?",
                "How to position for optimal digging?"
            ],
            "navigation": [
                "Which direction should the loader move?",
                "What's the safest approach path?",
                "How to navigate to the material pile?",
                "Where to position the loader?"
            ]
        }
        
        self.command_templates = [
            "Drive to the nearest pile and fill the bucket",
            "Position loader for digging at the closest pile",
            "Navigate to material pile and begin excavation",
            "Approach the pile from the best angle and dig",
            "Fill the bucket with material from the pile"
        ]
    
    def generate_vqa_pairs(self, image_path: str, pile_positions: List[PilePosition]) -> List[Dict]:
        """Generate VQA pairs for an image with detected piles"""
        vqa_pairs = []
        
        if not pile_positions:
            # No piles detected - generate negative examples
            vqa_pairs.append({
                "image": image_path,
                "question": "Where should the loader go?",
                "answer": "No material piles detected. Search the area for construction materials."
            })
            return vqa_pairs
        
        # Generate location-based questions
        nearest_pile = pile_positions[0]  # Already sorted by distance
        
        vqa_pairs.extend([
            {
                "image": image_path,
                "question": "Where is the nearest pile?",
                "answer": f"The nearest pile is at position ({nearest_pile.x:.0f}, {nearest_pile.y:.0f}) - {nearest_pile.label}"
            },
            {
                "image": image_path,
                "question": "What action should the loader take?",
                "answer": f"Drive to coordinates ({nearest_pile.x:.0f}, {nearest_pile.y:.0f}) and dig from the {nearest_pile.label}"
            },
            {
                "image": image_path,
                "question": "Fill the bucket with material",
                "answer": f"DRIVE_TO({nearest_pile.x:.0f}, {nearest_pile.y:.0f}); DIG_AT_POSITION({nearest_pile.x:.0f}, {nearest_pile.y:.0f})"
            }
        ])
        
        # Add template-based pairs
        for category, questions in self.question_templates.items():
            question = random.choice(questions)
            answer = self._generate_contextual_answer(question, pile_positions)
            
            vqa_pairs.append({
                "image": image_path,
                "question": question,
                "answer": answer
            })
        
        return vqa_pairs
    
    def _generate_contextual_answer(self, question: str, pile_positions: List[PilePosition]) -> str:
        """Generate contextual answers based on detected piles"""
        if not pile_positions:
            return "No piles detected. Search for construction materials."
        
        nearest = pile_positions[0]
        
        if "where" in question.lower() or "direction" in question.lower():
            return f"Move to position ({nearest.x:.0f}, {nearest.y:.0f}) where {nearest.label} is located"
        elif "action" in question.lower() or "approach" in question.lower():
            return f"Drive to the {nearest.label} at ({nearest.x:.0f}, {nearest.y:.0f}) and position for digging"
        else:
            return f"Target the {nearest.label} at coordinates ({nearest.x:.0f}, {nearest.y:.0f})"
