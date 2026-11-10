"""Lightweight smoke tests for the package's model/data contracts."""
import numpy as np
import torch

from synphen_cl.config import N_LABELS
from synphen_cl.embeddings.phenotypes import PHENOTYPE_LABELS
from synphen_cl.model import _build_model


def test_canonical_phenotype_order_and_count():
    assert len(PHENOTYPE_LABELS) == N_LABELS == 24
    assert PHENOTYPE_LABELS[0] == "Benign"
    assert PHENOTYPE_LABELS[1] == "Cellular"


def test_model_forward_shapes():
    label_prior = np.random.default_rng(0).normal(size=(N_LABELS, 1268)).astype(np.float32)
    model = _build_model(d_in=8, label_prior=label_prior, n_outputs=N_LABELS, device="cpu")
    model.eval()
    with torch.no_grad():
        logits_tab, logits_label, sample_repr, label_prototypes = model(torch.zeros(2, 8))
    assert tuple(logits_tab.shape) == (2, N_LABELS)
    assert tuple(logits_label.shape) == (2, N_LABELS)
    assert tuple(sample_repr.shape) == (2, 128)
    assert tuple(label_prototypes.shape) == (N_LABELS, 128)
