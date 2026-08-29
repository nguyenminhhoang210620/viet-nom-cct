"""Hàm mất mát entropy chéo có ràng buộc tập ứng viên (Eq 3.16-3.19)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .dataset import IGNORE_INDEX


def cct_masked_loss(scores: torch.Tensor, local_idx: torch.Tensor) -> torch.Tensor:
    """scores: (B,N,C) đã masked_fill(-inf) tại ô đệm ứng viên (không hợp lệ).
    local_idx: (B,N) chỉ số ứng viên đúng trong C(xi); IGNORE_INDEX cho vị trí không
    tham gia hàm mất mát (đệm câu, OOV, hoặc nhãn đúng không nằm trong C(xi))."""
    B, N, C = scores.shape
    logp = F.log_softmax(scores, dim=-1)
    loss = F.nll_loss(
        logp.reshape(B * N, C),
        local_idx.reshape(B * N),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    return loss


def token_accuracy(scores: torch.Tensor, local_idx: torch.Tensor) -> tuple[int, int]:
    """Trả về (số đúng, số vị trí hợp lệ) để tính Acctoken (Eq 4.3) tích luỹ qua nhiều batch."""
    valid = local_idx != IGNORE_INDEX
    if valid.sum().item() == 0:
        return 0, 0
    pred = scores.argmax(dim=-1)
    correct = ((pred == local_idx) & valid).sum().item()
    return int(correct), int(valid.sum().item())
