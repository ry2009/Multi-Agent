"""Few-shot examples for different question types."""

FEW_SHOT_EXAMPLES = {
    'arithmetic': [
        {
            'question': "What is 25 + 13?",
            'answer': "38",
            'explanation': "Adding 25 and 13: 25 + 13 = 38"
        },
        {
            'question': "Calculate 48 divided by 6",
            'answer': "8",
            'explanation': "Dividing 48 by 6: 48 ÷ 6 = 8"
        }
    ],
    'factual': [
        {
            'question': "What is the capital of Italy?",
            'answer': "Rome",
            'explanation': "Rome is the capital city of Italy."
        },
        {
            'question': "Who wrote 'Hamlet'?",
            'answer': "William Shakespeare",
            'explanation': "The play 'Hamlet' was written by William Shakespeare."
        }
    ],
    'logical': [
        {
            'question': "If all mammals are warm-blooded, and whales are mammals, are whales warm-blooded?",
            'answer': "Yes, whales are warm-blooded",
            'explanation': "Since all mammals are warm-blooded (premise 1), and whales are mammals (premise 2), we can conclude that whales are warm-blooded."
        },
        {
            'question': "If it's sunny, Sarah goes to the park. Sarah is at the park. Was it necessarily sunny?",
            'answer': "No, not necessarily",
            'explanation': "While we know that sunny weather leads to Sarah going to the park, her being at the park doesn't necessarily mean it was sunny. There could be other reasons for her to be there."
        }
    ],
    'word_problem': [
        {
            'question': "If a car travels at 40 mph for 3 hours, how far does it go?",
            'answer': "120 miles",
            'explanation': "Distance = Speed × Time\n40 mph × 3 hours = 120 miles"
        },
        {
            'question': "If bananas cost $0.25 each and apples cost $0.40 each, how much do 3 bananas and 2 apples cost?",
            'answer': "$1.55",
            'explanation': "Bananas: 3 × $0.25 = $0.75\nApples: 2 × $0.40 = $0.80\nTotal: $0.75 + $0.80 = $1.55"
        }
    ],
    'open_ended': [
        {
            'question': "What are the benefits of renewable energy?",
            'answer': "Renewable energy offers several key benefits: 1) Environmental protection through reduced carbon emissions, 2) Sustainable and infinite resource availability, 3) Lower long-term costs after initial investment, 4) Energy independence for countries and regions.",
            'explanation': "This answer provides specific points with clear reasoning and examples."
        },
        {
            'question': "How does technology affect modern education?",
            'answer': "Technology impacts modern education in multiple ways: 1) Increased accessibility through online learning, 2) Interactive and personalized learning experiences, 3) Improved communication between teachers and students, 4) Access to vast educational resources.",
            'explanation': "This answer structures the response with clear points and specific examples."
        }
    ]
}

def get_few_shot_prompt(question_type: str, num_examples: int = 2) -> str:
    """Generate a few-shot prompt for a given question type."""
    examples = FEW_SHOT_EXAMPLES.get(question_type, [])
    if not examples:
        return ""
    
    prompt = f"Here are some example {question_type} questions and their answers:\n\n"
    
    for i, example in enumerate(examples[:num_examples], 1):
        prompt += f"Example {i}:\n"
        prompt += f"Question: {example['question']}\n"
        prompt += f"Answer: {example['answer']}\n"
        prompt += f"Explanation: {example['explanation']}\n\n"
    
    return prompt 