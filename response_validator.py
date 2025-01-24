"""Response validation and quality filtering."""
import re
from typing import Tuple

class ResponseValidator:
    def __init__(self):
        self.min_length = 5  # Minimum response length in words
        self.max_length = 100  # Maximum response length in words
        
    def validate_arithmetic(self, response: str) -> Tuple[bool, str]:
        """Validate arithmetic responses."""
        # Try to extract a number
        numbers = re.findall(r'-?\d*\.?\d+', response)
        if not numbers:
            return False, "No numerical answer found"
        
        # Check if the answer is reasonably sized
        try:
            number = float(numbers[0])
            if abs(number) > 1e6:
                return False, "Answer is unreasonably large"
        except:
            return False, "Could not parse numerical answer"
            
        return True, numbers[0]
    
    def validate_factual(self, response: str) -> Tuple[bool, str]:
        """Validate factual responses."""
        words = response.split()
        
        if len(words) < self.min_length:
            return False, "Response too short"
        if len(words) > self.max_length:
            return False, "Response too long"
            
        # Check for question-like patterns (response shouldn't be a question)
        if '?' in response:
            return False, "Response contains questions"
            
        return True, response.strip()
    
    def validate_logical(self, response: str) -> Tuple[bool, str]:
        """Validate logical reasoning responses."""
        words = response.split()
        
        if len(words) < self.min_length:
            return False, "Response too short"
            
        # Look for logical connectives
        logical_markers = ['because', 'therefore', 'if', 'then', 'since', 'thus']
        has_logical_structure = any(marker in response.lower() for marker in logical_markers)
        
        if not has_logical_structure:
            return False, "No logical structure found"
            
        return True, response.strip()
    
    def validate_word_problem(self, response: str) -> Tuple[bool, str]:
        """Validate word problem responses."""
        # Should contain both numbers and units
        numbers = re.findall(r'-?\d*\.?\d+', response)
        units = re.findall(r'\b(dollars|miles|hours|kg|meters|feet|pounds|mph)\b', response.lower())
        
        if not numbers:
            return False, "No numerical answer found"
        if not units:
            return False, "No units found in answer"
            
        return True, response.strip()
    
    def validate_open_ended(self, response: str) -> Tuple[bool, str]:
        """Validate open-ended responses."""
        words = response.split()
        
        if len(words) < self.min_length:
            return False, "Response too short"
        if len(words) > self.max_length:
            return False, "Response too long"
            
        # Check for multiple points/arguments
        sentences = response.split('.')
        if len(sentences) < 2:
            return False, "Response lacks multiple points"
            
        return True, response.strip()
    
    def validate_response(self, response: str, question_type: str) -> Tuple[bool, str]:
        """Validate a response based on its question type."""
        # Basic validation
        if not response or not response.strip():
            return False, "Empty response"
            
        # Remove the question from the response if it was repeated
        if "Question:" in response:
            response = response.split("Question:")[-1]
        if "Answer:" in response:
            response = response.split("Answer:")[-1]
            
        response = response.strip()
        
        # Type-specific validation
        validation_methods = {
            'arithmetic': self.validate_arithmetic,
            'factual': self.validate_factual,
            'logical': self.validate_logical,
            'word_problem': self.validate_word_problem,
            'open_ended': self.validate_open_ended
        }
        
        validator = validation_methods.get(question_type, self.validate_factual)
        return validator(response) 