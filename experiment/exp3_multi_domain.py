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



@torch.no_grad()
def embed_texts(texts: List[str],
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                batch_size=64,
                normalize=True) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    if normalize:
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    return embs.cpu().numpy()


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
    # Embedding Diversity removed
    # embs = embed_texts(texts, model_name=embedding_model_name, batch_size=batch_size)

    # Self-BLEU
    self_bleu = compute_self_bleu(texts, sample_size=selfbleu_sample_size)
    out = {
        "num_texts": len(texts),
        "self_bleu": self_bleu
    }

    return out



def analyze_stealth_success_by_category(json_path: str) -> Dict[str, Tuple[int, float]]:
    """
    Count the number and percentage of stealth_success=True for each category
    
    Args:
        json_path: JSON file path
        
    Returns:
        Dictionary in format {category: (count, percentage)}
        Example: {"finance": (50, 25.5), "medical": (30, 15.3)}
    """
    # 1. Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. Count stealth_success=True for each category
    category_counts = {}
    total_success = 0
    
    for item in data:
        category = item.get("category")
        stealth_success = item.get("stealth_success")
        
        # Only count stealth_success=True
        if stealth_success is True:
            total_success += 1
            category_counts[category] = category_counts.get(category, 0) + 1
    
    # 3. Calculate percentage for each category
    result = {}
    for category, count in category_counts.items():
        percentage = (count / total_success * 100) if total_success > 0 else 0.0
        result[category] = (count, round(percentage, 2))
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="finance")
    parser.add_argument("--origin_json_key", type=str, default="prompt")
    parser.add_argument("--stealth_json_key", type=str, default="stealth_prompt")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--selfbleu_sample", type=int, default=-1)
    args = parser.parse_args()

    root = osp.dirname(osp.abspath(__file__))
    data_root = osp.join(root, "exp3_dataset", args.domain)

    origin_data_path = osp.join(data_root, "step3_evaluated_prompts.json")

    stealth_data_prefix = f"step7_quality_checked_{args.domain}"
    stealth_data_path = None
    for fname in os.listdir(data_root):
        if fname.startswith(stealth_data_prefix) and fname.endswith(".json"):
            stealth_data_path = osp.join(data_root, fname)
            break
    if stealth_data_path is None:
        raise FileNotFoundError(f"No file starting with '{stealth_data_prefix}' found in {data_root}")

    origin_toxicity_score = compute_toxicity_score(origin_data_path)

    origin_texts = read_texts(origin_data_path, args.origin_json_key)
    stealth_texts = read_steal_success_texts(stealth_data_path, args.stealth_json_key)

    origin_metric = compute_emb_div_and_self_bleu(texts=origin_texts, embedding_model_name=args.model, batch_size=args.batch_size, selfbleu_sample_size=args.selfbleu_sample)
    stealth_metric = compute_emb_div_and_self_bleu(texts=stealth_texts, embedding_model_name=args.model, batch_size=args.batch_size, selfbleu_sample_size=args.selfbleu_sample)
    distribution_of_stealth_success = analyze_stealth_success_by_category(stealth_data_path)    
    results = {
        "domain": args.domain,
        "toxicity_score": origin_toxicity_score,
        "origin_metric": origin_metric,
        "stealth_metric": stealth_metric,
        "distribution_of_stealth_success": distribution_of_stealth_success
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))

    result_dir = osp.join(root, "exp3_result")
    os.makedirs(result_dir, exist_ok=True)
    result_filename = args.domain + ".json"
    result_path = osp.join(result_dir, result_filename)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()