import argparse
import json
import numpy as np
import torch
from typing import List
from tqdm import tqdm
import os
import os.path as osp
from sentence_transformers import SentenceTransformer
from sacrebleu.metrics.bleu import BLEU


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="RiskAtlas_medicine_prompts_sampled.json")
    parser.add_argument("--json_key", type=str, default="prompt_text")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--selfbleu_sample", type=int, default=-1)
    args = parser.parse_args()

    root = osp.dirname(osp.abspath(__file__))
    data_path = osp.join(root, "exp4_dataset", "kg_ablation", args.input)

    texts = read_texts(data_path, )

    # Embedding Diversity removed
    embs = embed_texts(texts, model_name=args.model, batch_size=args.batch_size)

    # Self-BLEU
    self_bleu = compute_self_bleu(texts, sample_size=args.selfbleu_sample)

    out = {
        "num_texts": len(texts),
        "self_bleu": self_bleu
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    result_dir = osp.join(root, "exp4_result")
    os.makedirs(result_dir, exist_ok=True)
    result_filename = osp.splitext(args.input)[0] + ".json"
    result_path = osp.join(result_dir, result_filename)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()