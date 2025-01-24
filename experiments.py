from main import MultiAgentSystem, run_experiment
import json
from typing import List, Dict
import numpy as np
from tqdm import tqdm

# Comprehensive test cases covering different question types
TEST_CASES = {
    "arithmetic": [
        {"question": "What is 15 + 28?", "answer": "43"},
        {"question": "Calculate 125 divided by 5", "answer": "25"},
        {"question": "What is the square root of 144?", "answer": "12"},
        {"question": "What is 30% of 200?", "answer": "60"},
    ],
    "factual": [
        {"question": "What is the capital of Japan?", "answer": "Tokyo"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
        {"question": "What is the chemical symbol for gold?", "answer": "Au"},
        {"question": "Which planet is closest to the Sun?", "answer": "Mercury"},
    ],
    "logical": [
        {
            "question": "If all birds can fly, and penguins are birds, can penguins fly?",
            "answer": "No, because while penguins are birds, not all birds can fly. This is a false premise."
        },
        {
            "question": "If it's raining, the ground is wet. The ground is wet. Does this mean it's raining?",
            "answer": "No, because there could be other reasons why the ground is wet."
        },
    ],
    "word_problem": [
        {
            "question": "If a store sells apples for $0.50 each and oranges for $0.75 each, how much would 4 apples and 3 oranges cost?",
            "answer": "4.25"
        },
        {
            "question": "A train travels at 80 mph. How far will it travel in 2.5 hours?",
            "answer": "200 miles"
        },
    ],
    "open_ended": [
        {
            "question": "What are the potential implications of artificial intelligence on society?",
            "answer": None  # Open-ended questions don't have single correct answers
        },
        {
            "question": "How might climate change affect global agriculture in the next 50 years?",
            "answer": None
        },
    ]
}

def run_comprehensive_experiment(num_agents: int = 3, num_critics: int = 2) -> Dict:
    """Run experiments across all question types and analyze results."""
    system = MultiAgentSystem(num_agents=num_agents, num_critics=num_critics)
    all_results = {}
    
    print("\nInitializing Multi-Agent System...")
    print(f"Number of Agents: {num_agents}")
    print(f"Number of Critics: {num_critics}\n")
    
    for q_type, cases in tqdm(TEST_CASES.items(), desc="Question Types"):
        try:
            print(f"\n{'='*20} Testing {q_type.upper()} Questions {'='*20}")
            questions = [case["question"] for case in cases]
            answers = [case["answer"] for case in cases]
            
            results = []
            for q, a in tqdm(zip(questions, answers), desc=f"{q_type} Questions", total=len(questions)):
                try:
                    result = run_experiment(system, [q], [q_type], [a])
                    results.extend(result)
                    
                    # Print agent specialization development
                    print("\nAgent Specialization Development:")
                    for agent in system.agents:
                        stats = agent.get_performance_stats(q_type)
                        print(f"\n{agent.id}:")
                        print(f"  Specialization Score: {agent.get_specialization(q_type):.3f}")
                        print(f"  Performance Trend: {'↑' if stats['trend'] > 0 else '↓' if stats['trend'] < 0 else '→'}")
                        print(f"  Mean Performance: {stats['mean']:.3f} ± {stats['std']:.3f}")
                        
                except Exception as e:
                    print(f"Error processing question: {q}")
                    print(f"Error details: {str(e)}")
                    continue
            
            all_results[q_type] = results
            
        except Exception as e:
            print(f"Error processing question type: {q_type}")
            print(f"Error details: {str(e)}")
            continue
    
    return all_results

def analyze_results(results: Dict) -> Dict:
    """Analyze experimental results and compute metrics."""
    analysis = {}
    
    for q_type, type_results in results.items():
        type_analysis = {
            "num_questions": len(type_results),
            "avg_diversity": {
                "semantic": np.mean([r["diversity"]["semantic"] for r in type_results]),
                "lexical": np.mean([r["diversity"]["lexical"] for r in type_results]),
                "cluster": np.mean([r["diversity"]["cluster"] for r in type_results])
            },
            "diversity_trend": {
                "semantic": np.polyfit(range(len(type_results)), 
                    [r["diversity"]["semantic"] for r in type_results], 1)[0] if len(type_results) > 1 else 0.0,
                "lexical": np.polyfit(range(len(type_results)), 
                    [r["diversity"]["lexical"] for r in type_results], 1)[0] if len(type_results) > 1 else 0.0,
            }
        }
        
        # Calculate accuracy and confidence
        if any(r["correct_answer"] for r in type_results):
            correct_responses = []
            for r in type_results:
                if r["correct_answer"]:
                    is_correct = r["final_answer"].lower() == r["correct_answer"].lower()
                    correct_responses.append(is_correct)
            
            type_analysis["accuracy"] = np.mean(correct_responses)
            type_analysis["accuracy_std"] = np.std(correct_responses) if len(correct_responses) > 1 else 0.0
            
            # Calculate accuracy trend
            if len(correct_responses) > 1:
                type_analysis["accuracy_trend"] = np.polyfit(range(len(correct_responses)), 
                    correct_responses, 1)[0]
            else:
                type_analysis["accuracy_trend"] = 0.0
        
        analysis[q_type] = type_analysis
    
    return analysis

def print_detailed_analysis(analysis: Dict):
    """Print detailed analysis with trends and statistical significance."""
    print("\n=== Detailed Analysis ===")
    
    for q_type, metrics in analysis.items():
        print(f"\n{q_type.upper()} Questions:")
        print(f"Number of questions: {metrics['num_questions']}")
        
        print("\nDiversity Metrics:")
        for metric, score in metrics['avg_diversity'].items():
            trend = metrics['diversity_trend'].get(metric, 0.0)
            trend_symbol = "↑" if trend > 0.01 else "↓" if trend < -0.01 else "→"
            print(f"  {metric}: {score:.3f} {trend_symbol}")
        
        if "accuracy" in metrics:
            confidence_interval = 1.96 * metrics["accuracy_std"] / np.sqrt(metrics["num_questions"])
            trend_symbol = "↑" if metrics.get("accuracy_trend", 0) > 0.01 else "↓" if metrics.get("accuracy_trend", 0) < -0.01 else "→"
            
            print(f"\nAccuracy: {metrics['accuracy']:.2%} ± {confidence_interval:.2%} {trend_symbol}")
            print(f"Standard Deviation: {metrics['accuracy_std']:.3f}")

def save_results(results: Dict, filename: str = "experiment_results.json"):
    """Save experimental results to a JSON file."""
    # Convert results to JSON-serializable format
    serializable_results = {}
    for q_type, type_results in results.items():
        serializable_results[q_type] = []
        for result in type_results:
            serializable_result = {
                'question': result['question'],
                'type': result['type'],
                'correct_answer': result['correct_answer'],
                'final_answer': result['final_answer'],
                'diversity': {
                    k: float(v) if isinstance(v, np.float32) else v
                    for k, v in result['diversity'].items()
                },
                'responses': [
                    {'agent_id': agent_id, 'response': response}
                    for agent_id, response in result['responses']
                ]
            }
            serializable_results[q_type].append(serializable_result)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)

if __name__ == "__main__":
    try:
        # Run comprehensive experiments
        print("Starting Multi-Agent Experiments...")
        results = run_comprehensive_experiment(num_agents=3, num_critics=2)
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = analyze_results(results)
        
        # Print detailed analysis
        print_detailed_analysis(analysis)
        
        # Save results
        save_results(results)
        print("\nResults saved to experiment_results.json")
        
    except Exception as e:
        print(f"\nAn error occurred during the experiment:")
        print(f"Error details: {str(e)}")
        raise 