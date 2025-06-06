from typing import List, Tuple
from pile_detector import PilePosition

class ActionCommandGenerator:
    """Convert LLM responses to actionable robot commands"""
    
    def __init__(self):
        self.action_mappings = {
            "drive": "DRIVE_TO({x}, {y})",
            "dig": "DIG_AT_POSITION({x}, {y})",
            "position": "POSITION_BUCKET({angle})",
            "approach": "APPROACH_PILE({direction})",
            "stop": "STOP()",
            "backup": "BACKUP({distance})",
            "turn": "TURN({angle})"
        }
    
    def text_to_commands(self, text_response: str, pile_positions: List[PilePosition]) -> List[str]:
        """Convert text response to executable commands"""
        commands = []
        text_lower = text_response.lower()
        
        # Extract coordinates if present
        coords = self._extract_coordinates(text_response)
        
        # Parse actions from text
        if "drive" in text_lower or "move" in text_lower or "go" in text_lower:
            if coords:
                commands.append(f"DRIVE_TO({coords[0]}, {coords[1]})")
            elif pile_positions:
                target = pile_positions[0]
                commands.append(f"DRIVE_TO({target.x:.0f}, {target.y:.0f})")
        
        if "dig" in text_lower or "excavate" in text_lower or "fill" in text_lower:
            if coords:
                commands.append(f"DIG_AT_POSITION({coords[0]}, {coords[1]})")
            elif pile_positions:
                target = pile_positions[0]
                commands.append(f"DIG_AT_POSITION({target.x:.0f}, {target.y:.0f})")
        
        if "position" in text_lower or "approach" in text_lower:
            commands.append("POSITION_BUCKET(30)")  # Default angle
        
        if "stop" in text_lower:
            commands.append("STOP()")
        
        return commands if commands else ["NO_ACTION()"]
    
    def _extract_coordinates(self, text: str) -> Tuple[float, float] or None:
        """Extract coordinate pairs from text"""
        import re
        
        # Look for patterns like (123, 456) or coordinates 123, 456
        coord_pattern = r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)'
        matches = re.findall(coord_pattern, text)
        
        if matches:
            return (float(matches[0][0]), float(matches[0][1]))
        
        return None