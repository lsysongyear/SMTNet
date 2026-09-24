"""
改进的 word2vec + bert 特征融合模块
提供多种融合策略来替代简单的拼接（concatenation）

融合策略:
1. gated_fusion: 门控融合 - 学习可学习的门控权重动态平衡两种特征
2. cross_attention: 交叉注意力融合 - 使用交叉注意力让两种特征互相增强
3. proj_fusion: 低维投影融合 - 先将高维特征投影到低维共享空间再融合
4. multitask: 多任务学习 - 同时预测 word2vec 和 bert 两个目标

CLIP 对比学习改进:
- 注意力池化 (attention pooling) 替代均值池化
- MoCo 动量队列 (momentum queue) 提供大量负样本
- 可学习 temperature (logit_scale)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    门控融合 (Gated Fusion)
    学习一个门控权重来动态平衡 word2vec 和 bert 的贡献
    输出维度 = hidden_dim
    """
    def __init__(self, dim1=300, dim2=768, hidden_dim=256):
        super().__init__()
        self.proj1 = nn.Linear(dim1, hidden_dim, bias=False)
        self.proj2 = nn.Linear(dim2, hidden_dim, bias=False)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, w2v, bert):
        h1 = self.proj1(w2v)
        h2 = self.proj2(bert)
        gate_input = torch.cat([h1, h2], dim=-1)
        gate = self.gate_net(gate_input)
        fused = gate * h1 + (1 - gate) * h2
        fused = self.fusion_proj(fused)
        return fused


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合 (Cross-Attention Fusion)
    使用多头交叉注意力让两种特征互相增强
    输出维度 = hidden_dim
    """
    def __init__(self, dim1=300, dim2=768, hidden_dim=256, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj1 = nn.Linear(dim1, hidden_dim, bias=False)
        self.proj2 = nn.Linear(dim2, hidden_dim, bias=False)
        self.cross_attn1 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

    def forward(self, w2v, bert):
        if w2v.dim() == 2:
            w2v = w2v.unsqueeze(1)
            bert = bert.unsqueeze(1)
            squeeze_dim = True
        else:
            squeeze_dim = False

        h1 = self.proj1(w2v)
        h2 = self.proj2(bert)
        attn1_out, _ = self.cross_attn1(h1, h2, h2)
        attn2_out, _ = self.cross_attn2(h2, h1, h1)
        fused = torch.cat([attn1_out, attn2_out], dim=-1)
        fused = self.fusion(fused)

        if squeeze_dim:
            fused = fused.squeeze(1)
        return fused


class ProjFusion(nn.Module):
    """
    低维投影融合 (Project-then-Fuse)
    先将高维特征投影到低维共享空间，再进行融合
    输出维度 = hidden_dim
    """
    def __init__(self, dim1=300, dim2=768, hidden_dim=256):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Linear(dim1, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(dim2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

    def forward(self, w2v, bert):
        h1 = self.proj1(w2v)
        h2 = self.proj2(bert)
        concat = torch.cat([h1, h2], dim=-1)
        fused = self.fusion(concat)
        return fused


class MultiTaskHead(nn.Module):
    """
    多任务学习头
    共享编码器，分别预测 word2vec 和 bert
    """
    def __init__(self, hidden_dim=256, w2v_dim=300, bert_dim=768):
        super().__init__()
        self.w2v_head = nn.Linear(hidden_dim, w2v_dim, bias=False)
        self.bert_head = nn.Linear(hidden_dim, bert_dim, bias=False)

    def forward(self, x):
        w2v_pred = self.w2v_head(x)
        bert_pred = self.bert_head(x)
        return w2v_pred, bert_pred


def _strip_bias(module):
    """Recursively set bias=None on all Linear, Conv1d, and LayerNorm layers."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            if m.bias is not None:
                m.bias = None
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                m.bias = None


class CenteredActivation(nn.Module):
    """Anti-symmetric activation: f(x) = act(x) - act(-x).

    Guarantees E[f(x)] = 0 for any zero-mean distribution,
    preventing DC shift accumulation that causes representation collapse.
    """
    def __init__(self, base_act='gelu'):
        super().__init__()
        if base_act == 'gelu':
            self.act = nn.GELU()
        elif base_act == 'relu':
            self.act = nn.ReLU()
        elif base_act == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif base_act == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f'Unknown base activation: {base_act}')

    def forward(self, x):
        return self.act(x) - self.act(-x)


def _replace_activations(module):
    """Recursively replace GELU/ReLU/LeakyReLU/SiLU with CenteredActivation."""
    _act_map = {nn.GELU: 'gelu', nn.ReLU: 'relu',
                nn.LeakyReLU: 'leaky_relu', nn.SiLU: 'silu'}
    for name, child in list(module.named_children()):
        for act_cls, act_name in _act_map.items():
            if isinstance(child, act_cls):
                setattr(module, name, CenteredActivation(act_name))
                break
        else:
            _replace_activations(child)


class FusionBrainNetwork(nn.Module):
    """
    融合脑网络 - CLIP 对比学习

    改进:
    - 注意力池化 (attention pooling) 替代均值池化
    - MoCo 动量队列提供大量负样本
    - 可学习 temperature (logit_scale)
    - bias=False: 所有 Linear/Conv1d 层无 bias，防止逐词归一化后
      mean_pool 导致所有样本输出趋同（collapse）
    """
    def __init__(self, brain_encoder,
                 fusion_type='gated_fusion',
                 w2v_dim=300,
                 bert_dim=768,
                 fusion_hidden_dim=256,
                 temprature=None,
                 reg_loss=None,
                 queue_size=4096,
                 momentum=0.999):
        super().__init__()

        self.fusion_type = fusion_type
        self.brain_encoder = brain_encoder
        self.w2v_dim = w2v_dim
        self.bert_dim = bert_dim

        # 选择融合策略
        self._single_feat = None
        if fusion_type == 'w2v_only':
            self.fusion = None
            self.output_dim = w2v_dim
            self._single_feat = 'w2v'
        elif fusion_type == 'bert_only':
            self.fusion = None
            self.output_dim = bert_dim
            self._single_feat = 'bert'
        elif fusion_type == 'gated_fusion':
            self.fusion = GatedFusion(dim1=w2v_dim, dim2=bert_dim, hidden_dim=fusion_hidden_dim)
            self.output_dim = fusion_hidden_dim
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(dim1=w2v_dim, dim2=bert_dim, hidden_dim=fusion_hidden_dim)
            self.output_dim = fusion_hidden_dim
        elif fusion_type == 'proj_fusion':
            self.fusion = ProjFusion(dim1=w2v_dim, dim2=bert_dim, hidden_dim=fusion_hidden_dim)
            self.output_dim = fusion_hidden_dim
        elif fusion_type == 'multitask':
            self.fusion = None
            self.output_dim = w2v_dim + bert_dim
            self.multitask_head = MultiTaskHead(
                hidden_dim=fusion_hidden_dim,
                w2v_dim=w2v_dim,
                bert_dim=bert_dim
            )
        elif fusion_type == 'concat':
            self.fusion = None
            self.output_dim = w2v_dim + bert_dim
        else:
            raise ValueError(f'Unknown fusion type: {fusion_type}')

        self._modify_brain_encoder_output(self.output_dim)

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # 注意力池化 —— 替代 mean(dim=1)（bias=False 防止统一注意力权重）
        self.attn_pool = nn.Linear(self.output_dim, 1, bias=False)

        # 回归损失
        if reg_loss is not None:
            if reg_loss == 'mse':
                self.reg_loss = self._mse_loss
            elif reg_loss == 'corr':
                self.reg_loss = self._corr_loss
            else:
                raise ValueError('unknown reg_loss type')

        # CLIP 对比学习
        self.temprature = temprature
        if temprature is not None:
            # 预测侧投影头（bias=False 防止 collapse）
            self.pred_proj = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim, bias=False),
                nn.LayerNorm(self.output_dim),
                nn.GELU(),
                nn.Linear(self.output_dim, self.output_dim, bias=False),
            )
            # 目标侧投影头（在线）
            self.target_proj = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim, bias=False),
                nn.LayerNorm(self.output_dim),
                nn.GELU(),
                nn.Linear(self.output_dim, self.output_dim, bias=False),
            )
            # 目标侧投影头（动量，用于队列）
            self.target_proj_k = copy.deepcopy(self.target_proj)
            for param in self.target_proj_k.parameters():
                param.requires_grad = False

            # 可学习 temperature: 初始化为 1/temprature 的对数
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temprature))

        # MoCo 队列
        self.queue_size = queue_size
        self.momentum = momentum
        self.register_buffer('queue', F.normalize(torch.randn(queue_size, self.output_dim), dim=-1))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # Final: strip bias + replace activations to prevent representation collapse
        # NOTE: commenting out because these destroy the brain encoder's pretrained architecture
        # and prevent any learning (CenteredActivation outputs 0 for zero-mean inputs)
        # _strip_bias(self)
        # _replace_activations(self)

    def _modify_brain_encoder_output(self, new_dim):
        if hasattr(self.brain_encoder, 'final_linear'):
            in_features = self.brain_encoder.final_linear.in_features
            self.brain_encoder.final_linear = nn.Linear(in_features, new_dim, bias=False)

    @torch.no_grad()
    def _momentum_update(self):
        """动量更新 target_proj_k"""
        for p_online, p_momentum in zip(self.target_proj.parameters(),
                                         self.target_proj_k.parameters()):
            p_momentum.data = p_momentum.data * self.momentum + p_online.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """将当前 batch 的 key 入队"""
        if self.queue_size == 0:
            return
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
        else:
            self.queue[ptr:ptr + batch_size] = keys

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def _pool_temporal(self, pred):
        """注意力池化: [batch, seq_len, dim] -> [batch, dim]"""
        attn = self.attn_pool(pred)          # [batch, seq_len, 1]
        attn = F.softmax(attn, dim=1)        # [batch, seq_len, 1]
        pooled = (pred * attn).sum(dim=1)    # [batch, dim]
        return pooled

    def _mse_loss(self, pred, target):
        return nn.MSELoss(reduction='none')(pred, target).mean(dim=(1, 2))

    def _corr_loss(self, pred, target):
        from model import pearson_batch
        return 1 - pearson_batch(pred, target, dim=1).mean(dim=-1)

    def predict(self, x, sub_id):
        return self.brain_encoder(x, sub_id)

    def forward_fusion(self, w2v_embed, bert_embed):
        if self._single_feat == 'w2v':
            return w2v_embed
        elif self._single_feat == 'bert':
            return bert_embed
        elif self.fusion_type == 'multitask':
            return None
        elif self.fusion_type == 'concat':
            return torch.cat([w2v_embed, bert_embed], dim=-1)
        else:
            return self.fusion(w2v_embed, bert_embed)

    def forward_fusion_batch(self, w2v_batch, bert_batch):
        if self._single_feat == 'w2v':
            return w2v_batch
        elif self._single_feat == 'bert':
            return bert_batch
        elif self.fusion_type == 'multitask':
            return None
        elif self.fusion_type == 'concat':
            return torch.cat([w2v_batch, bert_batch], dim=-1)
        else:
            return self.fusion(w2v_batch, bert_batch)

    def get_reg_loss_batch(self, x, sub_id, w2v_embed, bert_embed):
        pred = self.predict(x, sub_id)

        if self.fusion_type == 'multitask':
            w2v_pred, bert_pred = self.multitask_head(pred)
            w2v_pred = w2v_pred.mean(dim=1)
            bert_pred = bert_pred.mean(dim=1)
            w2v_loss = self.reg_loss(w2v_pred.unsqueeze(-1), w2v_embed.unsqueeze(-1))
            bert_loss = self.reg_loss(bert_pred.unsqueeze(-1), bert_embed.unsqueeze(-1))
            total_loss = w2v_loss + bert_loss
            from model import pearson_batch
            concat_pred = torch.cat([w2v_pred, bert_pred], dim=-1)
            concat_target = torch.cat([w2v_embed, bert_embed], dim=-1)
            corr = pearson_batch(concat_pred.unsqueeze(-1), concat_target.unsqueeze(-1), dim=1).squeeze(-1).detach()
            return concat_pred, total_loss, corr
        else:
            target = self.forward_fusion(w2v_embed, bert_embed)
            pred_mean = pred.mean(dim=1)
            from model import pearson_batch
            corr = pearson_batch(pred_mean.unsqueeze(-1), target.unsqueeze(-1), dim=1).squeeze(-1).detach()
            loss = self.reg_loss(pred_mean.unsqueeze(-1), target.unsqueeze(-1))
            return pred_mean, loss, corr

    def _cosine_sim(self, a, b):
        a_norm = F.normalize(a, p=2, dim=-1)
        b_norm = F.normalize(b, p=2, dim=-1)
        return torch.matmul(a_norm, b_norm.T)

    def get_clip_loss_batch(self, x, sub_id, w2v_embed, bert_embed):
        """
        CLIP 对比学习损失 (改进版)
        - 注意力池化替代均值
        - MoCo 队列增强负样本
        - 可学习 temperature
        """
        pred = self.predict(x, sub_id)                     # [batch, seq_len, output_dim]
        pred_pooled = self._pool_temporal(pred)            # [batch, output_dim]

        target = self.forward_fusion_batch(w2v_embed, bert_embed)  # [batch, output_dim]

        # 在线投影
        pred_proj = self.pred_proj(pred_pooled)            # [batch, output_dim]
        target_proj = self.target_proj(target)              # [batch, output_dim]

        # 动量更新 + 动量投影 + 入队
        with torch.no_grad():
            self._momentum_update()
            target_k = self.target_proj_k(target)
            target_k = F.normalize(target_k, p=2, dim=-1)

        # 获取队列（所有已入队的 key）
        queue = self.queue.clone()

        loss, rank_acc = self._info_nce_loss(pred_proj, target_proj, queue)

        # 当前 batch 的 key 入队
        self._dequeue_and_enqueue(target_k)

        return pred_pooled, loss, rank_acc

    def _info_nce_loss(self, pred, target, queue):
        """
        InfoNCE 损失 (改进版)
        pred:   [batch, dim]  — 预测侧
        target: [batch, dim]  — 目标侧（在线），正样本在 [0..batch-1]
        queue:  [K, dim]      — 动量队列，全是负样本
        """
        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        # batch 内相似度: [batch, batch] — 对角线是正样本
        sim_inbatch = torch.matmul(pred, target.T)

        # 队列相似度: [batch, K] — 全是负样本
        sim_queue = torch.matmul(pred, queue.T)

        # 拼接: [batch, batch + K]
        sim = torch.cat([sim_inbatch, sim_queue], dim=-1)

        # 可学习 temperature
        sim = sim * self.logit_scale.exp()

        # 对角线是正样本
        loss = -self.logsoftmax(sim).diag()

        # Rank accuracy (仅在 batch 内排名)
        label = torch.arange(sim_inbatch.shape[0], device=sim.device)
        _, idx = sim_inbatch.sort(dim=1, descending=True)
        rank = torch.where(idx == label.unsqueeze(1))[1]
        rank_acc = (sim_inbatch.shape[1] - 1 - rank) / (sim_inbatch.shape[1] - 1)

        return loss, rank_acc

    def predict_label(self, x, sub_id, candidate_embeds, return_sim=False):
        """
        候选检索 (Top-1 / Top-10)
        """
        pred = self.predict(x, sub_id)
        pred_pooled = self._pool_temporal(pred)
        pred_proj = self.pred_proj(pred_pooled)

        sim = self._cosine_sim(pred_proj, candidate_embeds)

        _, top1 = torch.topk(sim, k=1, dim=-1)
        _, top10 = torch.topk(sim, k=10, dim=-1)

        if return_sim:
            return top1, top10, sim
        return top1, top10

    def prepare_candidates(self, w2v_all, bert_all, batch_size=1024, device=None):
        """
        预计算候选嵌入（融合 + 在线 target_proj 投影）
        """
        if device is None:
            device = next(self.parameters()).device

        all_proj = []
        self.eval()
        with torch.no_grad():
            for i in range(0, len(w2v_all), batch_size):
                w2v_b = w2v_all[i:i + batch_size].to(device)
                bert_b = bert_all[i:i + batch_size].to(device)
                fused = self.forward_fusion_batch(w2v_b, bert_b)
                proj = self.target_proj(fused)
                all_proj.append(proj.detach().cpu())
        return torch.cat(all_proj, dim=0)
