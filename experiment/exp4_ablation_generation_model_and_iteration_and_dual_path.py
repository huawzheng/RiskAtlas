import argparse
import json
import numpy as np
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm
import os
import os.path as osp
from sentence_transformers import SentenceTransformer
from sacrebleu.metrics.bleu import BLEU
from pathlib import Path

def compute_toxicity_score(path: str, json_key: str = "toxicity_evaluation") -> float:
    scores = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for obj in data:
                eval_obj = obj.get(json_key, {})
                scores.append(float(eval_obj["toxicity_score"]))
    else:
        raise ValueError("Unsupported file type. Use .json")
    if len(scores) < 1:
        raise ValueError("No toxicity scores found.")
    return sum(scores) / len(scores)


def read_texts(path: str, json_key: str = "prompt_text") -> List[str]:
    texts = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for obj in data:
                t = str(obj.get(json_key, "")).strip()
                if t:
                    texts.append(t)
    else:
        raise ValueError("Unsupported file type. Use .json")
    if len(texts) < 2:
        raise ValueError("Need at least 2 texts.")
    return texts



def read_steal_success_texts(path: str, json_key: str = "stealth_prompt") -> List[str]:
    texts = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for obj in data:
                flag = obj.get("stealth_success", False)
                if flag:
                    t = str(obj.get(json_key, "")).strip()
                    if t:
                        texts.append(t)
    else:
        raise ValueError("Unsupported file type. Use .json")
    if len(texts) < 2:
        raise ValueError("Need at least 2 texts.")
    return texts


def compute_self_bleu(texts: List[str],
                      sample_size: int = -1,
                      seed: int = 42,
                      max_ngram: int = 4,
                      tokenization: str = "13a") -> float:
    N = len(texts)
    rng = np.random.default_rng(seed)
    indices = list(range(N)) if sample_size < 0 or sample_size >= N else list(rng.choice(N, size=sample_size, replace=False))
    total = 0.0
    bleu = BLEU(
        max_ngram_order=max_ngram,
        effective_order=True,
        tokenize=tokenization
    )
    for i in tqdm(indices, desc="Self-BLEU"):
        hyp = texts[i]
        refs = texts[:i] + texts[i+1:]
        score = bleu.sentence_score(hyp, refs)
        total += score.score
    return total / len(indices)


def compute_emb_div_and_self_bleu(
    texts: List[str],
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=64,
    selfbleu_sample_size=-1
):
    # Self-BLEU
    self_bleu = compute_self_bleu(texts, sample_size=selfbleu_sample_size)
    out = {
        "num_texts": len(texts),
        "self_bleu": self_bleu
    }

    return out


def calculate_metrics(json_path: str, raw_total_count: int) -> Dict[str, float]:
    """
    Read JSON file and calculate OSR, avg_iter, cos_sim, ppl metrics.
    
    Args:
        json_path: JSON file path
        
    Returns:
        Dictionary containing calculated results:
        {
            "OSR": float,      # Obfuscated Success Rate (0-100)
            "avg_iter": float, # Average iterations for successful items
            "cos_sim": float,  # Average semantic similarity
            "ppl": float       # Average perplexity
        }
    """
    # 1. Read JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return {}
        
    if not data:
        return {"OSR": 0.0, "avg_iter": 0.0, "cos_sim": 0.0, "ppl": 0.0}

    total_count = len(data)
    success_items = []
    
    total_similarity = 0.0
    total_ppl = 0.0
    
    # 2. Iterate through data
    for item in data:
        # Extract fields, use get to prevent KeyError, default to 0 or False
        iteration = item.get("iterations", 0)
        ppl_score = item.get("ppl_score_stealth", 0.0)
        sim_score = item.get("similarity_score_stealth", 0.0)
        is_success = item.get("stealth_success", False)
        
        # Accumulate for cos_sim and ppl calculation (all items)
        total_similarity += sim_score
        total_ppl += ppl_score
        
        # Record successful items for OSR and avg_iter calculation
        if is_success:
            success_items.append(iteration)
            
    # 3. Calculate metrics
    # OSR: Success Rate (percentage)
    osr = (len(success_items) / raw_total_count * 100) if raw_total_count > 0 else 0.0
    
    # avg_iter: Average Iterations (only successful items)
    avg_iter = (sum(success_items) / len(success_items)) if success_items else 0.0
    
    # cos_sim: Average Similarity (all items)
    cos_sim = (total_similarity / total_count) if total_count > 0 else 0.0
    
    # ppl: Average Perplexity (all items)
    ppl = (total_ppl / total_count) if total_count > 0 else 0.0
    
    return {
        "success_count": len(success_items),
        "OSR": round(osr, 2),
        "avg_iter": round(avg_iter, 2),
        "cos_sim": round(cos_sim, 4),
        "ppl": round(ppl, 2)
    }





def calculate_similarity_and_ppl(json_path: str) -> Dict[str, float]:
    """
    Read JSON file and calculate OSR, avg_iter, cos_sim, ppl metrics.
    
    Args:
        json_path: JSON file path
        
    Returns:
        Dictionary containing calculated results:
        {
            "total_count": int, # Total sample count
            "cos_sim": float,  # Average semantic similarity
            "ppl": float       # Average perplexity
        }
    """
    # 1. Read JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return {}
        
    if not data:
        return {"OSR": 0.0, "avg_iter": 0.0, "cos_sim": 0.0, "ppl": 0.0}

    total_count = len(data)
    total_similarity = 0.0
    total_ppl = 0.0
    
    # 2. Iterate through data
    for item in data:
        # Extract fields, use get to prevent KeyError, default to 0 or False
        ppl_score = item.get("ppl_score_stealth", 0.0)
        sim_score = item.get("similarity_score_stealth", 0.0)
        
        # Accumulate for cos_sim and ppl calculation (all items)
        total_similarity += sim_score
        total_ppl += ppl_score
            
    # 3. Calculate metrics
    # cos_sim: Average Similarity (all items)
    cos_sim = (total_similarity / total_count) if total_count > 0 else 0.0
    
    # ppl: Average Perplexity (all items)
    ppl = (total_ppl / total_count) if total_count > 0 else 0.0
    
    return {
        "total_count": total_count,
        "cos_sim": round(cos_sim, 4),
        "ppl": round(ppl, 2)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_name", type=str, default="llama_70b_iter_18_dual_path")
    parser.add_argument("--origin_json_key", type=str, default="prompt")
    parser.add_argument("--stealth_json_key", type=str, default="stealth_prompt")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--selfbleu_sample", type=int, default=-1)
    args = parser.parse_args()

    root = osp.dirname(osp.abspath(__file__))
    data_root = osp.join(root, "exp4_dataset", args.ablation_name)

    # origin_data_path = osp.join(data_root, "step3_evaluated_prompts.json")
    # origin_toxicity_score = compute_toxicity_score(origin_data_path)
    # origin_texts = read_texts(origin_data_path, args.origin_json_key)


    stealth_data_prefix = f"step7_quality_checked_medicine"
    stealth_data_path = None
    for fname in os.listdir(data_root):
        if fname.startswith(stealth_data_prefix) and fname.endswith(".json"):
            stealth_data_path = osp.join(data_root, fname)
            break
    if stealth_data_path is None:
        raise FileNotFoundError(f"No file starting with '{stealth_data_prefix}' found in {data_root}")
    
    # stealth_texts = read_steal_success_texts(stealth_data_path, args.stealth_json_key)


    # origin_self_bleu = compute_emb_div_and_self_bleu(texts=origin_texts, embedding_model_name=args.model, batch_size=args.batch_size, selfbleu_sample_size=args.selfbleu_sample)
    # stealth_self_bleu = compute_emb_div_and_self_bleu(texts=stealth_texts, embedding_model_name=args.model, batch_size=args.batch_size, selfbleu_sample_size=args.selfbleu_sample)

    step6_data_prefix = f"final_huggingface_dataset"
    step6_data_path = None
    for fname in os.listdir(data_root):
        if fname.startswith(step6_data_prefix) and fname.endswith(".json"):
            step6_data_path = osp.join(data_root, fname)
            break
    if step6_data_path is None:
        raise FileNotFoundError(f"No file starting with '{step6_data_prefix}' found in {data_root}")

    step6_metrics = calculate_similarity_and_ppl(step6_data_path)
    raw_total_count = step6_metrics["total_count"]

    step7_metrics = calculate_metrics(stealth_data_path, raw_total_count)


    results = {
        "domain": args.ablation_name,
        # "toxicity_score": origin_toxicity_score,
        # "origin_self_bleu": origin_self_bleu,
        # "stealth_self_bleu": stealth_self_bleu,
        "step6_metrics": step6_metrics,
        "step7_metrics": step7_metrics
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))

    result_dir = osp.join(root, "exp4_result")
    os.makedirs(result_dir, exist_ok=True)
    result_filename = args.ablation_name + ".json"
    result_path = osp.join(result_dir, result_filename)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()