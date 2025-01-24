# Multi-Agent LLM Research

This repository contains research code exploring improvements to multi-agent language model systems, inspired by and building upon the work in ["Multiagent Finetuning of Language Models"](https://llm-multiagent-ft.github.io/).

## Key Improvements

- **Enhanced Specialization**: Implementation of dynamic agent specialization based on performance tracking
- **Sophisticated Critic System**: Weighted voting mechanism with specialized critic models
- **Diversity Measurement**: Semantic diversity scoring using sentence embeddings
- **Adaptive Learning**: Agents and critics that adapt their behavior based on historical performance

## Installation

```bash
# Clone the repository
git clone https://github.com/ry2009/Multi-Agent.git
cd Multi-Agent

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the main experiment:

```bash
python main.py
```

The system will:
1. Initialize multiple agents with different temperature settings
2. Create specialized critic models
3. Run debates on various question types
4. Track performance and adapt specializations
5. Output results including diversity scores and confidence levels

## Project Structure

- `main.py`: Core implementation of the multi-agent system
- `requirements.txt`: Project dependencies
- `.gitignore`: Git ignore rules

## Research Goals

This project aims to explore and potentially improve upon the specialization heuristic described in the original paper, specifically:

1. Testing if the specialization approach generalizes well across domains
2. Exploring the balance between maintaining diversity and ensuring convergence
3. Investigating scalability across different types of tasks

## Contributing

Feel free to open issues or submit pull requests with improvements or experiments. 