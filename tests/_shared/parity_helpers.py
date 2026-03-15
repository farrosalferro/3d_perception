"""Shared parity assertions for decode/postprocess checks."""

from __future__ import annotations

import torch


def assert_decoded_topk_label_score_consistency(
    cls_scores: torch.Tensor,
    decoded_scores: torch.Tensor,
    decoded_labels: torch.Tensor,
    *,
    max_num: int,
    num_classes: int,
) -> None:
    flat_scores = cls_scores.sigmoid().reshape(-1)
    topk = min(int(max_num), int(flat_scores.numel()))
    topk_scores, topk_indices = flat_scores.topk(topk)
    topk_labels = (topk_indices % num_classes).to(dtype=torch.long)

    assert decoded_scores.ndim == 1
    assert decoded_labels.ndim == 1
    assert decoded_scores.shape[0] == decoded_labels.shape[0]
    assert decoded_scores.shape[0] <= topk
    if decoded_scores.numel() > 1:
        assert torch.all(decoded_scores[:-1] >= decoded_scores[1:])
    assert torch.all((decoded_scores >= 0.0) & (decoded_scores <= 1.0))
    if decoded_labels.numel() > 0:
        assert decoded_labels.dtype == torch.long
        assert int(decoded_labels.min().item()) >= 0
        assert int(decoded_labels.max().item()) < num_classes

    remaining = [(float(score), int(label)) for score, label in zip(topk_scores.tolist(), topk_labels.tolist())]
    for score, label in zip(decoded_scores.tolist(), decoded_labels.tolist()):
        matched_idx = None
        for idx, (candidate_score, candidate_label) in enumerate(remaining):
            if candidate_label == int(label) and abs(candidate_score - float(score)) <= 1e-6:
                matched_idx = idx
                break
        assert matched_idx is not None, "Decoded score/label pair is inconsistent with top-k logits."
        remaining.pop(matched_idx)

