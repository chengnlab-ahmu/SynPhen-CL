"""Generate CODER embeddings for the 24 phenotype labels (Benign + 23 HPO groups).

The CODER model files must be downloaded separately and placed in models/coder/.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ..config import CODER_MODEL_NAME, CODER_NPY, DEVICE
from .phenotypes import PHENOTYPE_DEFINITIONS, PHENOTYPE_LABELS

MAX_LEN = 64


def load_coder(model_dir=CODER_MODEL_NAME):
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError as exc:
        raise ImportError("CODER embedding generation requires `pip install transformers`.") from exc
    print(f"[INFO] Loading local CODER model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_terms(terms, tokenizer, model):
    enc = tokenizer(terms, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    out = model(**enc)
    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
    return F.normalize(hidden[:, 0, :], p=2, dim=1).cpu().numpy()


def embed_phenotype(entry, tokenizer, model):
    vectors = encode_terms(entry["terms"], tokenizer, model)
    pooled = vectors.mean(axis=0)
    return pooled / (np.linalg.norm(pooled) + 1e-12)


def main(model_dir=CODER_MODEL_NAME, output_path=CODER_NPY):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tokenizer, model = load_coder(model_dir)
    embeddings = np.stack([embed_phenotype(item, tokenizer, model) for item in tqdm(PHENOTYPE_DEFINITIONS, desc="CODER labels")]).astype(np.float32)
    np.save(output_path, embeddings)
    labels_path = os.path.splitext(output_path)[0] + "_labels.txt"
    with open(labels_path, "w", encoding="utf-8") as handle:
        for i, label in enumerate(PHENOTYPE_LABELS):
            handle.write(f"{i}\t{label}\n")
    print(f"[OK] embeddings: {output_path} shape={embeddings.shape}")
    print(f"[OK] labels: {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=CODER_MODEL_NAME)
    parser.add_argument("--output", default=CODER_NPY)
    args = parser.parse_args()
    main(args.model_dir, args.output)
