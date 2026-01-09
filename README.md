<h2 align="center">
RiskAtlas: Exposing Domain-Specific Risks in LLMs through Knowledge-Graph-Guided Harmful Prompt Generation
</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2601.04740">
    <img src="https://img.shields.io/badge/arXiv-2601.04740-b31b1b.svg" />
  </a>
</p>


## Abstract

Large language models (LLMs) are increasingly applied in specialized domains such as finance and healthcare, where they introduce unique safety risks. Domain-specific datasets of harmful prompts remain scarce and still largely rely on manual construction; public datasets mainly focus on explicit harmful prompts, which modern LLM defenses can often detect and refuse. In contrast, implicit harmful prompts—expressed through indirect domain knowledge—are harder to detect and better reflect real-world threats. We identify two challenges: transforming domain knowledge into actionable constraints and increasing the implicitness of generated harmful prompts. To address them, we propose an end-to-end framework that first performs knowledge-graph-guided harmful prompt generation to systematically produce domain-relevant prompts, and then applies dual-path obfuscation rewriting to convert explicit harmful prompts into implicit variants via direct and context-enhanced rewriting. This framework yields high-quality datasets combining strong domain relevance with implicitness, enabling more realistic red-teaming and advancing LLM safety research.

## Overview

### Pipeline

<p align="center">
  <img src="images/overview.svg" width="100%" />
</p>


### Core Features

- **Knowledge Graph Integration**: Leverages Wikidata for domain-specific entity extraction
- **Multi-Domain Support**: Configurable domains (medicine, finance, education, law)
- **Toxicity Evaluation**: Automated harmful content detection using Granite Guardian
- **Implicit Rewrite Generation**: Academic disguise and steganographic techniques
- **Complete Pipeline**: End-to-end workflow from knowledge extraction to evaluation

## System Architecture

<p align="center">
<pre>
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Knowledge     │    │   Harmful Prompt │    │   Toxicity      │    │   Dataset       │
│   Graph         │ -> │   Generation     │ -> │   Evaluation    │ -> │   Filtering     │
│   (Step 1)      │    │   (Step 2)       │    │   (Step 3)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
                                                                              v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Quality       │    │   Dataset        │    │   Implicit      │    │   Dataset       │
│   Check         │ <- │   Finalization   │ <- │   Rewrite       │ <- │   Assembly      │
│   (Step 7)      │    │   (Step 6)       │    │   (Step 5)      │    │   (Step 4)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
</pre>
</p>

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU
- Neo4j database
- 16GB+ memory

### Environment Setup

We recommend installing the environment using the `environment.yml` file.
```Bash
conda env create -f environment.yml
```
If the command above fails, you can also build your environment with follwling steps:
```bash
# create new environment
create -n RiskAtlas python=3.11
conda activate RiskAtlas

# Install pytorch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install packages
pip install -r requirements.txt
pip install --upgrade datasets
```

### Neo4j Prepration

3. Install and configure Neo4j database:

#### Method 1: Using Package Manager (Recommended)
```bash
# Ubuntu/Debian
sudo apt install neo4j

# Or using Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/12345678 neo4j:latest
```

#### Method 2: Tarball Installation
```bash
# 1. Download and extract Neo4j Community Edition
wget https://dist.neo4j.org/neo4j-community-5.x.x-unix.tar.gz
tar -zxvf neo4j-community-5.x.x-unix.tar.gz

# 2. Install OpenJDK 21
sudo apt install openjdk-21-jdk
# Or download and configure environment variables:
export JAVA_HOME=/path/to/jdk-21
export PATH=${JAVA_HOME}/bin:$PATH

# 3. Install required plugins
cd neo4j-community-5.x.x/plugins/
# Download neosemantics plugin (n10s)
wget https://github.com/neo4j-labs/neosemantics/releases/download/5.20.0/neosemantics-5.20.0.jar
# Download APOC plugin
wget https://github.com/neo4j/apoc/releases/download/5.x.x/apoc-5.x.x-core.jar

# 4. Start Neo4j
cd ../
./bin/neo4j-admin server console
```

#### Connection Configuration
```bash
# Update Neo4j credentials in configs/pipeline_config.yaml
# Default configuration:
# URI: bolt://localhost:7687
# Username: neo4j
```

## Model Prepration
Excute `instruct_finetune_code/llama_3_1_70b_alpaca_finetune.py` for obtaining `synthesis model` and `obfuscation model`.

Excute `instruct_finetune_code/llama_3_1_8b_alpaca_finetune.py` for obtaining `instrct finetune model`.

## Pipeline Running
**Remenber to start `neo4j` before running pipeline.**
```Bash
# build knowledge graph
python real_pipeline/step1_simple_kg_build.py --domain {your domain}

# load synthesis model and generate harmful prompts
bash scripts/start_vllm_finetune_server.sh
python real_pipeline/step2_harmful_generation.py --domain {your domain}
tmux kill-window -t llama3_1_70b_finetune

# load evaluate model and make toxicity evaluation
bash scripts/start_vllm_granite_guardian_server.sh
python real_pipeline/step3_toxicity_evaluation.py --domain {your domain}
tmux kill-window -t granite_guardian_evaluator

# filter prompts
python real_pipeline/step4_dataset_assembly.py --domain {your domain}

# load obfuscation model and make obfuscation on prompts
bash scripts/start_vllm_finetune_server.sh
python real_pipeline/step5_optimized_batch.py --domain {your domain}
tmux kill-window -t llama3_1_70b_finetune

# generate final domain dataset
python real_pipeline/step6_dataset_finalization.py --domain {your domain}

# quality check on final dataset using Gemini model
python real_pipeline/step7_quality_check.py --domain {your domain}
```

If you want to run the pipeline in one go, you can use the bash `run_pipeline.sh`.

```Bash
bash run_pipeline.sh --domain {your domain}
```

## Quick Start (Alternative Individual Steps)

### 1. Start Model Servers

```bash
# Start LLaMA fine-tuned model server
bash scripts/start_vllm_finetune_server.sh

# Start Granite Guardian evaluation server
bash scripts/start_vllm_granite_guardian_server.sh
```

### 2. Run Pipeline Steps

```bash
# Step 1: Build knowledge graph
python real_pipeline/step1_simple_kg_build.py --domain medicine

# Step 2: Generate harmful prompts
python real_pipeline/step2_harmful_generation.py --domain medicine

# Step 3: Evaluate toxicity
python real_pipeline/step3_toxicity_evaluation.py --domain medicine

# Step 4: Assemble filtered dataset
python real_pipeline/step4_dataset_assembly.py --domain medicine

# Step 5: Generate implicit attacks
python real_pipeline/step5_optimized_batch.py --domain medicine
# Alternative: python real_pipeline/step5_implicit_jailbreak.py --domain medicine

# Step 6: Dataset finalization
python real_pipeline/step6_dataset_finalization.py --domain medicine

# Step 7: Quality check
python real_pipeline/step7_quality_check.py --domain medicine
```

### 3. Complete Pipeline

```bash
# Run complete pipeline for medicine domain
bash run_pipeline.sh
```

## Configuration

### Domain Configuration

Configure domain-specific parameters in `configs/domains/{domain}.yaml`:

```yaml
domain:
  name: "medicine"
  description: "Medical domain knowledge graph"
  ...
```

### Model Configuration

Update model endpoints in `configs/models.yaml`:

```yaml
harmful_prompt_generator:
  model_name: "llama3.1-70b-finetune-vllm"
  server_url: "http://localhost:8000/v1"

toxicity_evaluator:
  model_name: "granite-guardian-3.1-8b-vllm-server"
  server_url: "http://localhost:8001/v1"
```

## Pipeline Stages

1. **Knowledge Graph Construction** (`step1`): Extract domain entities from Wikidata
2. **Harmful Prompt Generation** (`step2`): Generate harmful prompts using fine-tuned LLM
3. **Toxicity Evaluation** (`step3`): Evaluate generated content using safety model
4. **Dataset Assembly** (`step4`): Filter and compile high-quality dataset
5. **Implicit Jailbreak** (`step5`): Create disguised adversarial prompts
6. **Dataset Finalization** (`step6`): Final processing and quality control
7. **Quality Check** (`step7`): Validate dataset quality using Gemini model

## Output Structure

```
real_pipeline/outputs/{domain}/
├── step1_kg_entities.json          # Extracted entities
├── step2_harmful_prompts.json      # Generated prompts
├── step3_toxicity_scores.json      # Evaluation results
├── step4_filtered_dataset.json     # Curated dataset
├── step5_implicit_attacks.json     # Disguised prompts
├── step6_final_dataset.json        # Final processed dataset
└── step7_quality_checked.json      # Quality-validated dataset
```

## Evaluation Metrics

- **Toxicity Score**: Harmful content probability (0-1)
- **Attack Success Rate**: Percentage of successful adversarial prompts
- **Fluency Score**: Language quality assessment
- **Diversity Metrics**: Prompt variation analysis

## Research Applications

- Adversarial prompt generation and detection
- AI safety evaluation framework
- Knowledge-enhanced attack methodology
- Multi-domain harmful content analysis

## Test

### Experiment directory tree
```Text
--experiment--exp1_exp2_dataset # datasets of public benchmarks and our random selected data
           |--exp3_dataset--{domain name}--final*.json # RiskAtlas final dataset
           |                            |--step3*.json # original dataset with toxicity score
           |--exp3_result # evaluation results for multi-domain experiments
           |--exp4_dataset--RiskAtlas_medicine_prompts_sampled.json
           |             |--without_kg_step2_data.json
           |--exp4_result # ablation study results
           |--ASRDemoEvaluator.py # ASR evaluation utility
           |--exp1_dataset.py # code for experiment 1
           |--exp2_finetune.py # finetune code for experiment 2
           |--exp2_evaluate.py # evaluation for experiment 2
           |--exp3_multi_domain.py # code for experiment 3
           |--exp4_ablation_kg.py # code for experiment 4 (knowledge graph ablation)
           |--exp4_ablation_generation_model_and_iteration_and_dual_path.py # code for experiment 4 (model & iteration ablation)
```


### Experiment 1: Evaluation of ASR on public benchmarks and our RiskAtlas

```Bash
python experiment/exp1_dataset.py
```

### Experiment 2: safety finetune
model finetune
```Bash

python experiment/exp2_finetune.py --finetune_dataset {finetune dataset neme}
```
The safety finetuned model will be saved in `experiment/finetune_model`.

Then fill in the path parameters and run the bash `scripts/start_vllm_Llama-3.1-8B-finetune.sh` for loading safety finetuned model.

Lastly, evaluate the safety finetuned model on unseen datasets.
```Bash
python experiment/exp2_evaluate.py --model {safety finetuned model name} --dataset {attack dataset name} --finetune_dataset {finetune dataset name}
```

If you want to evaluate `LLMU` of the safety finetuned model.
```Bash
conda activate RiskAtlas

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness

pip install -e .
pip install --upgrade datasets

lm_eval --model hf --model_args pretrained={safety finetuned model path},parallelize=True,load_in_4bit=True,peft={LoRA weigh path} --tasks mmlu --num_fewshot 5 --device cuda:0 --batch_size 8 --output_path <your output path>
```

### Experiment 3: metric evaluation for multi domains
To evaluate toxicity score and diversity of the generated domain datasets:
```Bash
python experiment/exp3_multi_domain.py --domain {your domain}
```

### Experiment 4: comparison between the results with and w/o knowledge graph
```Bash
python experiment/exp4_ablation_kg.py
```
