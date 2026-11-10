"""Generate cui2vec phenotype embeddings from a local pretrained CUI vector CSV."""
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..config import CUI2VEC_SOURCE_CSV, CUI2VEC_NPY
from .phenotypes import PHENOTYPE_DEFINITIONS, PHENOTYPE_LABELS

EMB_DIM = 500


def load_cui2vec(path):
    print(f"[INFO] Loading cui2vec from: {path}")
    df = pd.read_csv(path, index_col=0)
    vectors = {str(idx): row.values.astype(np.float32) for idx, row in df.iterrows()}
    if vectors:
        actual_dim = len(next(iter(vectors.values())))
        if actual_dim != EMB_DIM:
            raise ValueError(f"Expected {EMB_DIM}-dimensional cui2vec vectors, found {actual_dim}")
    print(f"[INFO] Loaded {len(vectors):,} CUI vectors")
    return vectors


def embed_phenotype(entry, cui_dict):
    vectors = []
    seen = set()
    for cui in entry["cuis"]:
        if cui in seen:
            continue
        seen.add(cui)
        if cui in cui_dict:
            vectors.append(cui_dict[cui])
    if not vectors:
        print(f"[WARN] No CUI vector found for '{entry['label']}', using zero vector")
        return np.zeros(EMB_DIM, dtype=np.float32)
    pooled = np.stack(vectors).mean(axis=0)
    norm = np.linalg.norm(pooled)
    return (pooled / norm if norm > 1e-12 else pooled).astype(np.float32)


def main(source_path=CUI2VEC_SOURCE_CSV, output_path=CUI2VEC_NPY):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cui_dict = load_cui2vec(source_path)
    embeddings = np.stack([embed_phenotype(item, cui_dict) for item in tqdm(PHENOTYPE_DEFINITIONS, desc="cui2vec labels")]).astype(np.float32)
    np.save(output_path, embeddings)
    labels_path = os.path.splitext(output_path)[0] + "_labels.txt"
    with open(labels_path, "w", encoding="utf-8") as handle:
        for i, label in enumerate(PHENOTYPE_LABELS):
            handle.write(f"{i}\t{label}\n")
    print(f"[OK] embeddings: {output_path} shape={embeddings.shape}")
    print(f"[OK] labels: {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=CUI2VEC_SOURCE_CSV)
    parser.add_argument("--output", default=CUI2VEC_NPY)
    args = parser.parse_args()
    main(args.source, args.output)
