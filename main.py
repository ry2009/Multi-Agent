from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict, Counter
import json
import random

# Load GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class Agent:
    def __init__(self, id, temperature=1.0, specialization=None):
        self.id = id
        self.temperature = temperature
        self.specialization = specialization or []  # Types of questions this agent is good at
        self.correct_answers = defaultdict(int)  # Track correct answers by question type
        self.total_answers = defaultdict(int)  # Track total answers by question type

    def generate_response(self, prompt, max_length=50):
        """Generate a response with the agent's temperature."""
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=self.temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def update_performance(self, question_type, was_correct):
        """Update the agent's performance tracking."""
        self.total_answers[question_type] += 1
        if was_correct:
            self.correct_answers[question_type] += 1

    def get_accuracy(self, question_type):
        """Get the agent's accuracy for a specific question type."""
        if self.total_answers[question_type] == 0:
            return 0.0
        return self.correct_answers[question_type] / self.total_answers[question_type]

class CriticModel:
    def __init__(self, id, temperature=0.7):
        self.id = id
        self.temperature = temperature
        self.specialization = defaultdict(float)  # Track critic's expertise by question type

    def evaluate_response(self, question, response, question_type):
        """Evaluate a response and return a confidence score."""
        prompt = f"Question: {question}\nResponse: {response}\nIs this response correct? Rate from 0-1:"
        evaluation = model.generate(
            tokenizer(prompt, return_tensors="pt").input_ids,
            max_length=20,
            temperature=self.temperature
        )
        # Convert the evaluation to a float between 0 and 1
        try:
            score = float(tokenizer.decode(evaluation[0], skip_special_tokens=True)[:3])
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5

    def update_specialization(self, question_type, accuracy):
        """Update the critic's specialization based on its performance."""
        self.specialization[question_type] = (
            self.specialization[question_type] * 0.9 + accuracy * 0.1
        )

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
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_response_diversity(self, responses):
        """Calculate diversity score for a set of responses."""
        if not responses:
            return 0.0
        embeddings = self.embedding_model.encode([r[1] for r in responses])
        similarity_matrix = cosine_similarity(embeddings)
        return 1 - np.mean(similarity_matrix)

    def weighted_vote(self, responses, confidences):
        """Perform weighted voting based on critic confidences."""
        if not responses:
            return None, 0
        
        weighted_votes = defaultdict(float)
        for (_, response), confidence in zip(responses, confidences):
            weighted_votes[response] += confidence
        
        best_response = max(weighted_votes.items(), key=lambda x: x[1])
        return best_response

    def debate(self, question, question_type):
        """Run a multi-agent debate on a question."""
        # Generate responses from agents
        responses = []
        for agent in self.agents:
            if question_type in agent.specialization or not agent.specialization:
                response = agent.generate_response(question)
                responses.append((agent.id, response))

        # Calculate diversity
        diversity = self.get_response_diversity(responses)

        # Get critic evaluations
        confidences = []
        for response in responses:
            critic_scores = []
            for critic in self.critics:
                if question_type in critic.specialization or not critic.specialization:
                    score = critic.evaluate_response(question, response[1], question_type)
                    critic_scores.append(score)
            confidences.append(np.mean(critic_scores) if critic_scores else 0.5)

        # Get final answer through weighted voting
        final_answer, confidence = self.weighted_vote(responses, confidences)

        return {
            'responses': responses,
            'diversity': diversity,
            'final_answer': final_answer,
            'confidence': confidence
        }

def run_experiment(system, questions, question_types):
    """Run experiments on a set of questions."""
    results = []
    for question, q_type in zip(questions, question_types):
        result = system.debate(question, q_type)
        results.append({
            'question': question,
            'type': q_type,
            'diversity': result['diversity'],
            'responses': result['responses'],
            'final_answer': result['final_answer'],
            'confidence': result['confidence']
        })
    return results

if __name__ == "__main__":
    # Example questions with different types
    questions = [
        "What is 12 + 7?",
        "What is the capital of France?",
        "If a train travels at 60 mph for 2 hours, how far does it go?"
    ]
    question_types = ["arithmetic", "factual", "word_problem"]

    # Initialize system
    system = MultiAgentSystem(num_agents=3, num_critics=2)
    
    # Run experiment
    results = run_experiment(system, questions, question_types)
    
    # Print results
    for result in results:
        print(f"\nQuestion: {result['question']}")
        print(f"Type: {result['type']}")
        print(f"Diversity Score: {result['diversity']:.3f}")
        print("Responses:")
        for agent_id, response in result['responses']:
            print(f"  {agent_id}: {response}")
        print(f"Final Answer: {result['final_answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
