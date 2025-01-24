from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
import json
import random
from question_types import get_question_type, QUESTION_TYPES
from diversity_metrics import DiversityMetrics

# Load GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class Agent:
    def __init__(self, id, temperature=1.0):
        self.id = id
        self.temperature = temperature
        self.specialization_scores = defaultdict(float)
        self.response_history = defaultdict(list)

    def generate_response(self, prompt, question_type, max_length=50):
        """Generate a response with the agent's temperature."""
        # Add specialization context to the prompt
        context = f"You are specialized in {question_type} questions. "
        full_prompt = context + prompt
        
        inputs = tokenizer(full_prompt, return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=self.temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Track response
        self.response_history[question_type].append(response)
        return response

    def update_specialization(self, question_type: str, score: float):
        """Update specialization score for a question type."""
        # Exponential moving average
        alpha = 0.1
        current = self.specialization_scores[question_type]
        self.specialization_scores[question_type] = (1 - alpha) * current + alpha * score

    def get_specialization(self, question_type: str) -> float:
        """Get specialization score for a question type."""
        return self.specialization_scores[question_type]

class CriticModel:
    def __init__(self, id, temperature=0.7):
        self.id = id
        self.temperature = temperature
        self.evaluation_history = defaultdict(list)
        self.diversity_metrics = DiversityMetrics()

    def evaluate_response(self, question: str, response: str, question_type: str, correct_answer: str = None) -> dict:
        """Evaluate a response and return detailed metrics."""
        # Get question type evaluator
        q_type = get_question_type(question_type)
        
        # Calculate base score
        base_score = q_type.evaluate(response, correct_answer) if correct_answer else 0.5
        
        # Calculate confidence based on historical performance with this question type
        confidence = self.get_confidence(question_type)
        
        # Track evaluation
        self.evaluation_history[question_type].append(base_score)
        
        return {
            'score': base_score,
            'confidence': confidence,
            'question_type': question_type
        }

    def get_confidence(self, question_type: str) -> float:
        """Calculate confidence for a question type based on historical performance."""
        history = self.evaluation_history[question_type]
        if not history:
            return 0.5
        return sum(history) / len(history)

class MultiAgentSystem:
    def __init__(self, num_agents=3, num_critics=2):
        self.agents = [
            Agent(f"Agent_{i}", temperature=0.7 + (i * 0.3))
            for i in range(num_agents)
        ]
        self.critics = [
            CriticModel(f"Critic_{i}", temperature=0.7 + (i * 0.2))
            for i in range(num_critics)
        ]
        self.diversity_metrics = DiversityMetrics()

    def debate(self, question: str, question_type: str, correct_answer: str = None) -> dict:
        """Run a multi-agent debate on a question."""
        # Generate responses from agents
        responses = []
        for agent in self.agents:
            response = agent.generate_response(question, question_type)
            responses.append((agent.id, response))

        # Calculate diversity metrics
        response_texts = [r[1] for r in responses]
        diversity_scores = self.diversity_metrics.get_comprehensive_diversity(response_texts)

        # Get critic evaluations
        evaluations = []
        for response in responses:
            critic_scores = []
            for critic in self.critics:
                eval_result = critic.evaluate_response(
                    question, response[1], question_type, correct_answer
                )
                critic_scores.append(eval_result)
            evaluations.append(critic_scores)

        # Update agent specializations based on evaluations
        for (agent_id, response), agent_evals in zip(responses, evaluations):
            agent = next(a for a in self.agents if a.id == agent_id)
            avg_score = sum(e['score'] for e in agent_evals) / len(agent_evals)
            agent.update_specialization(question_type, avg_score)

        # Weighted voting for final answer
        final_answer = self.weighted_vote(responses, evaluations)

        return {
            'responses': responses,
            'diversity': diversity_scores,
            'evaluations': evaluations,
            'final_answer': final_answer
        }

    def weighted_vote(self, responses, evaluations) -> str:
        """Perform weighted voting based on critic evaluations and agent specialization."""
        weighted_votes = defaultdict(float)
        
        for (agent_id, response), agent_evals in zip(responses, evaluations):
            # Get agent's specialization score
            agent = next(a for a in self.agents if a.id == agent_id)
            specialization_weight = agent.get_specialization(agent_evals[0]['question_type'])
            
            # Calculate vote weight
            avg_score = sum(e['score'] * e['confidence'] for e in agent_evals) / len(agent_evals)
            vote_weight = avg_score * (1 + specialization_weight)
            
            weighted_votes[response] += vote_weight
        
        return max(weighted_votes.items(), key=lambda x: x[1])[0]

def run_experiment(system: MultiAgentSystem, questions: list, question_types: list, correct_answers: list = None):
    """Run experiments on a set of questions."""
    results = []
    
    if correct_answers is None:
        correct_answers = [None] * len(questions)
    
    for question, q_type, answer in zip(questions, question_types, correct_answers):
        result = system.debate(question, q_type, answer)
        results.append({
            'question': question,
            'type': q_type,
            'correct_answer': answer,
            'diversity': result['diversity'],
            'responses': result['responses'],
            'final_answer': result['final_answer']
        })
    return results

if __name__ == "__main__":
    # Example questions with different types and known answers
    test_cases = [
        {
            'question': "What is 12 + 7?",
            'type': "arithmetic",
            'answer': "19"
        },
        {
            'question': "What is the capital of France?",
            'type': "factual",
            'answer': "Paris"
        },
        {
            'question': "If a train travels at 60 mph for 2 hours, how far does it go?",
            'type': "word_problem",
            'answer': "120 miles"
        },
        {
            'question': "Why do objects fall towards the Earth?",
            'type': "logical",
            'answer': "Objects fall towards the Earth because of gravity"
        }
    ]
    
    # Initialize system
    system = MultiAgentSystem(num_agents=3, num_critics=2)
    
    # Run experiment
    results = run_experiment(
        system,
        [case['question'] for case in test_cases],
        [case['type'] for case in test_cases],
        [case['answer'] for case in test_cases]
    )
    
    # Print results
    for result in results:
        print(f"\nQuestion: {result['question']}")
        print(f"Type: {result['type']}")
        print(f"Correct Answer: {result['correct_answer']}")
        print(f"Diversity Metrics:")
        for metric, score in result['diversity'].items():
            print(f"  {metric}: {score:.3f}")
        print("Responses:")
        for agent_id, response in result['responses']:
            print(f"  {agent_id}: {response}")
        print(f"Final Answer: {result['final_answer']}")
