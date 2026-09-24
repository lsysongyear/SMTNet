"""Subject-indexed input projection around the unchanged BrainMagic backbone."""

from torch import nn

from .BrainNetwork import Brain_Magic_speech
from .public_baseline_utils import SubjectSpatialProjection


class SubjectProjectedBrainMagic(nn.Module):
    """Preserve the original shared MLP and all three temporal branches."""

    requires_subject_ids = True

    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=5, depthwise_kernel=15, dropout=0.1, n_subjects=1):
        super().__init__()
        self.spatial_projection = SubjectSpatialProjection(input_dim, n_subjects)
        self.model = Brain_Magic_speech(
            input_dim=input_dim, attention_dim=attention_dim, output_dim=output_dim,
            kernel_size=kernel_size, depthwise_kernel=depthwise_kernel,
            dropout=dropout,
        )

    def forward(self, x, subject_ids=None):
        return self.model(self.spatial_projection(x, subject_ids))
