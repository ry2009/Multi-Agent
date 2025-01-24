from typing import Dict, Any, List
import re
import numpy as np

class QuestionType:
    def __init__(self, name: str, evaluation_fn=None, difficulty: float = 1.0):
        self.name = name
        self.evaluation_fn = evaluation_fn or self.default_evaluation
        self.difficulty = difficulty
        self.performance_history = []

    def default_evaluation(self, response: str, correct_answer: str) -> float:
        """Default evaluation comparing exact matches."""
        return float(response.strip().lower() == correct_answer.strip().lower())

    def evaluate(self, response: str, correct_answer: str) -> float:
        """Evaluate a response and return a score between 0 and 1."""
        score = self.evaluation_fn(response, correct_answer)
        self.performance_history.append(score)
        return score

    def get_average_performance(self) -> float:
        """Get the average performance for this question type."""
        if not self.performance_history:
            return 0.0
        return np.mean(self.performance_history)

def arithmetic_evaluation(response: str, correct_answer: str) -> float:
    """Evaluate arithmetic responses with partial credit."""
    try:
        # Extract numbers from response and correct answer
        response_num = float(re.findall(r'-?\d*\.?\d+', response)[0])
        correct_num = float(re.findall(r'-?\d*\.?\d+', correct_answer)[0])
        
        # Calculate relative error
        error = abs(response_num - correct_num)
        if error == 0:
            return 1.0
        elif error <= 0.01 * abs(correct_num):  # 1% tolerance
            return 0.9
        elif error <= 0.1 * abs(correct_num):   # 10% tolerance
            return 0.5
        else:
            return 0.0
    except:
        return 0.0

def factual_evaluation(response: str, correct_answer: str) -> float:
    """Evaluate factual responses using keyword matching."""
    response_words = set(response.lower().split())
    answer_words = set(correct_answer.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(response_words.intersection(answer_words))
    union = len(response_words.union(answer_words))
    
    return intersection / union if union > 0 else 0.0

def logical_evaluation(response: str, correct_answer: str) -> float:
    """Evaluate logical reasoning responses."""
    # Look for key phrases indicating logical structure
    logic_markers = ['because', 'therefore', 'if', 'then', 'since', 'thus']
    has_logic_structure = any(marker in response.lower() for marker in logic_markers)
    
    # Basic answer correctness
    basic_score = float(response.strip().lower() == correct_answer.strip().lower())
    
    # Boost score if logical structure is present
    return min(1.0, basic_score + (0.2 if has_logic_structure else 0.0))

# Define standard question types
QUESTION_TYPES = {
    'arithmetic': QuestionType('arithmetic', arithmetic_evaluation, difficulty=1.0),
    'factual': QuestionType('factual', factual_evaluation, difficulty=0.8),
    'logical': QuestionType('logical', logical_evaluation, difficulty=1.2),
    'word_problem': QuestionType('word_problem', arithmetic_evaluation, difficulty=1.5),
    'open_ended': QuestionType('open_ended', None, difficulty=2.0),
}

def get_question_type(type_name: str) -> QuestionType:
    """Get a question type by name."""
    return QUESTION_TYPES.get(type_name, QuestionType(type_name))

def register_question_type(name: str, evaluation_fn=None, difficulty: float = 1.0):
    """Register a new question type."""
    QUESTION_TYPES[name] = QuestionType(name, evaluation_fn, difficulty) 