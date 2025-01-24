from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
import json
import random
from question_types import get_question_type, QUESTION_TYPES
from diversity_metrics import DiversityMetrics
from prompts.few_shot_examples import get_few_shot_prompt
from response_validator import ResponseValidator
import numpy as np

# Load GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set padding token to be the same as EOS token
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = model.config.eos_token_id

class Agent:
    def __init__(self, id, temperature=1.0):
        self.id = id
        self.temperature = temperature
        self.specialization_scores = defaultdict(float)
        self.response_history = defaultdict(list)
        self.performance_history = defaultdict(list)  # Track performance over time
        self.adaptation_rate = 0.1  # Base adaptation rate
        self.validator = ResponseValidator()
        self.consecutive_successes = defaultdict(int)
        self.success_threshold = 3  # Number of consecutive successes needed for specialization boost

    def generate_response(self, prompt, question_type, max_length=500, max_attempts=3):
        """Generate a response with retries for validation."""
        # Get few-shot examples
        few_shot = get_few_shot_prompt(question_type)
        
        # Create a more sophisticated prompt
        type_specific_instructions = {
            'arithmetic': "Solve this math problem step by step. Your final answer should be just the number.",
            'factual': "Answer this factual question directly and concisely. No explanations needed.",
            'logical': "Analyze this logical problem and give a clear yes/no answer with brief reasoning.",
            'word_problem': "Solve this word problem step by step. Your final answer should include units if applicable.",
            'open_ended': "Provide a clear, structured response with specific examples and reasoning."
        }
        
        instruction = type_specific_instructions.get(question_type, "Answer the following question:")
        
        # Construct the full prompt with examples and clear formatting
        full_prompt = (
            f"{instruction}\n\n"
            f"{few_shot}\n"
            f"Question: {prompt}\n"
            f"Think through this step by step:\n"
            f"1) First, understand what is being asked\n"
            f"2) Then, plan your approach\n"
            f"3) Finally, provide your answer\n\n"
            f"Answer: "
        )
        
        # Try generating valid responses
        for attempt in range(max_attempts):
            try:
                inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=max(0.2, self.temperature * (1.0 - self.specialization_scores[question_type])),
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
                
                # Extract only the generated response
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Clean up response
                response = response.strip()
                if response.startswith("Answer:"):
                    response = response[7:].strip()
                
                # Additional cleaning based on question type
                if question_type == 'arithmetic':
                    # Extract just the number for arithmetic
                    import re
                    numbers = re.findall(r'-?\d*\.?\d+', response)
                    if numbers:
                        response = numbers[-1]  # Take the last number as the final answer
                elif question_type == 'factual':
                    # Take first sentence for factual
                    response = response.split('.')[0].strip()
                
                # Validate response
                is_valid, cleaned_response = self.validator.validate_response(response, question_type)
                if is_valid:
                    self.response_history[question_type].append(cleaned_response)
                    return cleaned_response
                
            except Exception as e:
                print(f"Error in response generation (attempt {attempt + 1}): {str(e)}")
                continue
            
            # Adjust temperature for next attempt
            self.temperature *= 0.8
        
        # If all attempts fail, return the last response anyway
        return response

    def update_specialization(self, question_type: str, score: float):
        """Update specialization score with enhanced reinforcement."""
        # Track performance
        self.performance_history[question_type].append(score)
        
        # Update consecutive successes
        if score > 0.8:  # High success threshold
            self.consecutive_successes[question_type] += 1
        else:
            self.consecutive_successes[question_type] = 0
            
        # Calculate trend in recent performance
        recent_scores = self.performance_history[question_type][-5:]
        if len(recent_scores) >= 2:
            trend = sum(b - a for a, b in zip(recent_scores[:-1], recent_scores[1:])) / (len(recent_scores) - 1)
        else:
            trend = 0
            
        # Calculate specialization boost based on consecutive successes
        specialization_boost = min(0.2, 0.05 * self.consecutive_successes[question_type])
        
        # Adjust adaptation rate based on performance trend and specialization
        if trend > 0:
            effective_rate = self.adaptation_rate * (1 + trend + specialization_boost)
        else:
            effective_rate = self.adaptation_rate * (1 + trend/2)
            
        # Update specialization score with momentum and boost
        current = self.specialization_scores[question_type]
        momentum = 0.9
        
        new_score = (
            momentum * current +
            (1 - momentum) * (current + effective_rate * (score - current))
        )
        
        # Apply specialization boost if threshold met
        if self.consecutive_successes[question_type] >= self.success_threshold:
            new_score += specialization_boost
            
        # Update score with bounds
        self.specialization_scores[question_type] = max(0.0, min(1.0, new_score))

    def get_specialization(self, question_type: str) -> float:
        """Get specialization score with confidence and success bonus."""
        base_score = self.specialization_scores[question_type]
        
        # Calculate confidence based on number of responses
        n_responses = len(self.performance_history[question_type])
        confidence = min(1.0, n_responses / 10.0)
        
        # Add bonus for consecutive successes
        success_bonus = min(0.2, 0.05 * self.consecutive_successes[question_type])
        
        return min(1.0, base_score * confidence + success_bonus)

    def get_performance_stats(self, question_type: str) -> dict:
        """Get detailed performance statistics."""
        scores = self.performance_history[question_type]
        if not scores:
            return {
                "mean": 0.0,
                "std": 0.0,
                "trend": 0.0,
                "n_samples": 0,
                "consecutive_successes": self.consecutive_successes[question_type]
            }
            
        return {
            "mean": np.mean(scores),
            "std": np.std(scores) if len(scores) > 1 else 0.0,
            "trend": np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) > 1 else 0.0,
            "n_samples": len(scores),
            "consecutive_successes": self.consecutive_successes[question_type]
        }

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
