"""Subject projection shared by the two public speech-detection baselines."""

from numbers import Real

import torch
from torch import nn


def positive_integer(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def dropout_probability(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value < 1:
        raise ValueError(f"{name} must be a number in [0, 1)")


class SubjectSpatialProjection(nn.Module):
    """Independent C -> 4C -> C GELU projections, preserving sample order."""

    def __init__(self, input_dim, n_subjects):
        super().__init__()
        positive_integer("input_dim", input_dim)
        positive_integer("n_subjects", n_subjects)
        self.input_dim = input_dim
        self.n_subjects = n_subjects
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 4 * input_dim),
                nn.GELU(),
                nn.Linear(4 * input_dim, input_dim),
                nn.GELU(),
            )
            for _ in range(n_subjects)
        ])

    def forward(self, x, subject_ids=None):
        if x.ndim != 3 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected [B,{self.input_dim},T], got {tuple(x.shape)}")
        if x.shape[0] == 0 or x.shape[2] == 0:
            raise ValueError("Spatial projection requires a nonempty batch and time axis")
        if subject_ids is None:
            if self.n_subjects != 1:
                raise ValueError("subject_ids are required for a multisubject model")
            return self.projections[0](x.transpose(1, 2)).transpose(1, 2)
        if not isinstance(subject_ids, torch.Tensor):
            raise TypeError("subject_ids must be a one-dimensional integer tensor")
        if subject_ids.ndim != 1 or subject_ids.shape[0] != x.shape[0]:
            raise ValueError("subject_ids must have exactly one entry per batch sample")
        if subject_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("subject_ids must have an integer dtype")
        ids = subject_ids.to(device=x.device, dtype=torch.long)
        unique_ids = torch.unique(ids, sorted=True).tolist()
        if unique_ids[0] < 0 or unique_ids[-1] >= self.n_subjects:
            raise ValueError(f"subject_ids must be in [0, {self.n_subjects - 1}]")

        time_first = x.transpose(1, 2)
        if self.n_subjects == 1:
            return self.projections[0](time_first).transpose(1, 2)
        # Match existing baselines' per-example GEMM shapes, including TF32
        # behavior, instead of regrouping mixed-subject samples into new batches.
        output = [self.projections[int(ids[i])](time_first[i:i + 1])
                  for i in range(x.shape[0])]
        return torch.cat(output, dim=0).transpose(1, 2)
