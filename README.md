# Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains

A multi-agent system for improving language model performance through diverse reasoning and specialization.

## Project Structure

```
.
├── main.py              # Core multi-agent system implementation
├── experiments.py       # Experiment runner and analysis
├── prompts/            
│   └── few_shot_examples.py  # Few-shot examples for different question types
├── results/             # Experiment results directory
└── requirements.txt     # Project dependencies
```

## Setup

1. Create a Python environment:
```bash
conda create -n multiagent python=3.11
conda activate multiagent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

Run the main experiment suite:
```bash
python experiments.py
```

## Current Status

The system implements:
- Multi-agent debate with 3 agents and 2 critics
- 5 question types: arithmetic, factual, logical, word problems, and open-ended
- Diversity metrics: semantic, lexical, and clustering
- Specialization mechanism with performance tracking
- Response validation and quality filtering

Current challenges:
1. Low accuracy across question types (0% currently)
2. Limited specialization development
3. Response quality needs improvement
4. Model generation parameters need tuning

## Next Steps

1. Improve response generation:
   - Better prompt engineering
   - Fine-tune temperature and sampling parameters
   - Enhance response validation

2. Enhance specialization:
   - Adjust learning rates
   - Improve reinforcement mechanism
   - Better performance tracking

3. System improvements:
   - More sophisticated critic evaluation
   - Enhanced debate mechanism
   - Better result logging and analysis 