# ============================================================================
# average_groups.py - 通道分组平均层
# ============================================================================

from torch import nn
import torch


# ============================================================================
# AverageGroups - 将输入通道按组平均
# ============================================================================
class AverageGroups(nn.Module):
    """
    将输入通道划分为 n_groups 组，每组内取平均。

    输入: [B, C, L]  (batch, channels, sequence_length)
    输出: [B, n_groups, L]

    假设输入通道已按组排列：如 C = n_groups * channels_per_group，
    则本层将连续的 channels_per_group 个通道平均为一个通道。

    等效操作：
    x.reshape(B, n_groups, C//n_groups, L).mean(dim=2)
    """
    def __init__(self, n_groups):
        """
        Args:
            n_groups: 目标分组数（输出通道数），C 必须能被 n_groups 整除
        """
        super().__init__()
        self.n_groups = n_groups


    def forward(self, x):
        """
        Args:
            x: [batch_size, total_channels, seq_len]

        Returns:
            [batch_size, n_groups, seq_len] 分组平均后的张量
        """
        return x.view(x.size(0), self.n_groups, x.size(1)//self.n_groups, x.size(2)).mean(dim=1)


# ============================================================================
# 单元测试
# ============================================================================
def test_average_groups():
    """测试 AverageGroups 的正确性"""
    x = torch.randn(10, 100, 57)
    avg_groups = AverageGroups(10)
    # 100 通道分成 10 组，每组 10 通道 → 输出应为 [10, 10, 57]
    assert avg_groups(x).shape == (10, 10, 57)

    # 验证均值计算是否等于手动分组后求平均
    x_groups = [torch.randn(10, 10, 57) for _ in range(10)]
    x_concat = torch.cat(x_groups, dim=1)
    assert torch.all(avg_groups(x_concat) ==
                     torch.stack(x_groups, dim=1).mean(dim=1))


if __name__ == "__main__":
    test_average_groups()
    print("test successful")
