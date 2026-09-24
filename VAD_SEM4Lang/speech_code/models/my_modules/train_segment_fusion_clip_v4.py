"""
基于 train_segment_fusion_clip_v3.py — 加入 GatedFusion / CrossAttention / ProjFusion

与 v3 的区别:
  - 融合策略: FeatureFusion (v3) → GatedFusion / CrossAttention / ProjFusion (v4)
  - 新增 SegmentFusionNetwork 替代 BrainNetwork，支持时序维度的特征融合
  - 评估时预计算 fused candidates，避免逐 batch 重复融合
  - 新增 --fusion_type 和 --fusion_hidden_dim 参数
  - 其余保持一致: Pearson + InfoNCE + 12 被试联合训练 + 非重叠分段 + StepLR

使用方法:
    # word2vec+bert + gated fusion
    python train_segment_fusion_clip_v4.py \
        --feature_name word2vec_bert \
        --fusion_type gated_fusion \
        --device cuda:0

    # bert only (无融合)
    python train_segment_fusion_clip_v4.py \
        --feature_name bert \
        --device cuda:0
"""
import numpy as np
import config as cfg
import model_config as mcfg
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import SEM4Ldataset, get_EEG_SEM4L, get_feat_SEM4L
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
import time
import os
import os.path as op
import json
import pickle
import scipy.io
from scipy.signal import butter, filtfilt
from model import FeatureFusion, pearson_pair_wise, pearson_pair_wise_batch
from model_fusion import GatedFusion, CrossAttentionFusion, ProjFusion
from brain_decoder.awavenet import WaveNet
from brain_decoder.dilatedconv import DilatedConvNet
from brain_decoder.covconcatnet import ConvConcatNet
from brain_decoder.vlaai import VLAAI
from brain_decoder.vlaai2 import VLAAI2
from model_config import Config_VLAAI, Config_ConvConcatNet, Config_WaveNet, Config_DilatedConvNet


# 注册特征维度
mcfg.feature_dim_dict['bert'] = 768
mcfg.feature_dim_dict['word2vec'] = 300
mcfg.feature_dim_dict['bert_emb'] = 768  # alias

# BERT / word2vec 嵌入路径
_PROJECT_ROOT = '/mnt/Disk1/public/public/CASdata'
_BERT_DIR = op.join(_PROJECT_ROOT, 'metadata', 'stimuli', 'annotations',
                    'embeddings', 'bert', 'word-level')
_WORD2VEC_DIR = op.join(_PROJECT_ROOT, 'metadata', 'stimuli', 'annotations',
                        'embeddings', 'word2vec', 'word-level', '300d')
_TIME_ALIGN_DIR = op.join(_PROJECT_ROOT, 'metadata', 'stimuli', 'annotations',
                          'time_align', 'word-level')


def _lowpass_filter(data, fs, cutoff=4.0, order=4):
    """4Hz 低通滤波，与 GPT 处理 (mne filter_data 0-4Hz) 一致"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


# ===========================================================================
# BERT / word2vec 数据加载函数（与 v3 完全相同）
# ===========================================================================

def _load_bert_word_onset_and_embs(story_ids):
    onset_dict, offset_dict, bert_dict = {}, {}, {}
    for st in story_ids:
        story_name = f'story_{st}'
        wt_file = op.join(_TIME_ALIGN_DIR, f'story_{st}_word_time.mat')
        wt = scipy.io.loadmat(wt_file)
        starts = wt['start'].flatten().astype(np.float64) - 10.65
        ends = wt['end'].flatten().astype(np.float64) - 10.65
        bert_file = op.join(_BERT_DIR, f'story_{st}_word_bert_1-12_768.mat')
        data = scipy.io.loadmat(bert_file)['data']
        word_embs = data[11, :, :].astype(np.float32)
        onset_dict[story_name] = starts
        offset_dict[story_name] = ends
        bert_dict[story_name] = word_embs
    return onset_dict, offset_dict, bert_dict


def _load_word2vec_word_onset_and_embs(story_ids):
    onset_dict, offset_dict, w2v_dict = {}, {}, {}
    for st in story_ids:
        story_name = f'story_{st}'
        wt_file = op.join(_TIME_ALIGN_DIR, f'story_{st}_word_time.mat')
        wt = scipy.io.loadmat(wt_file)
        starts = wt['start'].flatten().astype(np.float64) - 10.65
        ends = wt['end'].flatten().astype(np.float64) - 10.65
        w2v_file = op.join(_WORD2VEC_DIR, f'story_{st}_word_word2vec.mat')
        data = scipy.io.loadmat(w2v_file)['data']
        word_embs = data.astype(np.float32)
        onset_dict[story_name] = starts
        offset_dict[story_name] = ends
        w2v_dict[story_name] = word_embs
    return onset_dict, offset_dict, w2v_dict


def get_feat_word2vec_SEM4L(seg_len, fs, use_lowpass=True):
    seg_len_samples = int(seg_len * fs)
    story_ids = list(range(1, 61))
    onset_dict, offset_dict, w2v_dict = _load_word2vec_word_onset_and_embs(story_ids)
    sample_sub = op.join(cfg.derivatives_dir, 'SEM4Lang', 'processed_data', 'sub-01')
    with open(op.join(sample_sub, 'sub-01-meg-std.pkl'), 'rb') as f:
        meg_ls = pickle.load(f)['MEG_ls']
    meg_lengths = {f'story_{st}': len(meg_ls[st - 1]) for st in story_ids}
    feat_dict = {}
    for st in story_ids:
        story_name = f'story_{st}'
        onsets = onset_dict[story_name]
        offsets = offset_dict[story_name]
        embs = w2v_dict[story_name]
        total_frames = meg_lengths[story_name]
        cont = np.zeros((total_frames, 300), dtype=np.float64)
        for i in range(len(onsets)):
            onset_fr = max(0, int(round(onsets[i] * fs)))
            offset_fr = min(total_frames, int(round(offsets[i] * fs)))
            if offset_fr > onset_fr:
                cont[onset_fr:offset_fr, :] = embs[i].astype(np.float64)
        if use_lowpass:
            cont = _lowpass_filter(cont, fs, cutoff=4.0).astype(np.float32)
        else:
            cont = cont.astype(np.float32)
        n_seg = total_frames // seg_len_samples
        cont = cont[:n_seg * seg_len_samples]
        feat_dict[story_name] = cont.reshape(n_seg, seg_len_samples, 300)
    feat_keys = sorted(feat_dict.keys())
    feat_frames = np.concatenate([feat_dict[k] for k in feat_keys], axis=0)
    feat_frames_id = np.arange(feat_frames.shape[0])
    feat_frames_num = [feat_dict[k].shape[0] for k in feat_keys]
    feat_frames_id_split = np.split(feat_frames_id, np.cumsum(feat_frames_num)[0:-1])
    feat_frames_id_dict = {k: v for k, v in zip(feat_keys, feat_frames_id_split)}
    return feat_dict, feat_frames_id_dict, feat_keys, feat_frames


def get_feat_bert_SEM4L(seg_len, fs, use_lowpass=True, sliding_window=False):
    seg_len_samples = int(seg_len * fs)
    story_ids = list(range(1, 61))
    onset_dict, offset_dict, bert_dict = _load_bert_word_onset_and_embs(story_ids)
    sample_sub = op.join(cfg.derivatives_dir, 'SEM4Lang', 'processed_data', 'sub-01')
    with open(op.join(sample_sub, 'sub-01-meg-std.pkl'), 'rb') as f:
        meg_ls = pickle.load(f)['MEG_ls']
    meg_lengths = {f'story_{st}': len(meg_ls[st - 1]) for st in story_ids}
    feat_dict = {}
    for st in story_ids:
        story_name = f'story_{st}'
        onsets = onset_dict[story_name]
        offsets = offset_dict[story_name]
        embs = bert_dict[story_name]
        total_frames = meg_lengths[story_name]
        cont = np.zeros((total_frames, 768), dtype=np.float64)
        for i in range(len(onsets)):
            onset_fr = max(0, int(round(onsets[i] * fs)))
            offset_fr = min(total_frames, int(round(offsets[i] * fs)))
            if offset_fr > onset_fr:
                cont[onset_fr:offset_fr, :] = embs[i].astype(np.float64)
        if use_lowpass:
            cont = _lowpass_filter(cont, fs, cutoff=4.0).astype(np.float32)
        else:
            cont = cont.astype(np.float32)
        n_seg = total_frames // seg_len_samples
        cont = cont[:n_seg * seg_len_samples]
        feat_dict[story_name] = cont.reshape(n_seg, seg_len_samples, 768)
    feat_keys = sorted(feat_dict.keys())
    feat_frames = np.concatenate([feat_dict[k] for k in feat_keys], axis=0)
    feat_frames_id = np.arange(feat_frames.shape[0])
    feat_frames_num = [feat_dict[k].shape[0] for k in feat_keys]
    feat_frames_id_split = np.split(feat_frames_id, np.cumsum(feat_frames_num)[0:-1])
    feat_frames_id_dict = {k: v for k, v in zip(feat_keys, feat_frames_id_split)}
    return feat_dict, feat_frames_id_dict, feat_keys, feat_frames


# ===========================================================================
# SegmentFusionNetwork — v4 核心模型
# ===========================================================================

class SegmentFusionNetwork(nn.Module):
    """段级别融合网络 — 支持 GatedFusion / CrossAttention / ProjFusion

    输入:
      - MEG 信号 [B, T, 204]
      - embed [B, T, total_dim] — word2vec+bert 沿特征维度拼接后的时序信号

    前向流程:
      1. 从 embed 中拆分出 word2vec [B, T, 300] 和 bert [B, T, 768]
      2. 目标侧融合: fusion(w2v, bert) → [B, T, feat_dim]
      3. 脑编码器: MEG → [B, T, feat_dim]
      4. Pearson pairwise InfoNCE loss
    """

    def __init__(self, brain_encoder, fusion_type='gated_fusion',
                 w2v_dim=300, bert_dim=768, fusion_hidden_dim=256,
                 temprature=None):
        super().__init__()
        self.brain_encoder = brain_encoder
        self.fusion_type = fusion_type
        self.w2v_dim = w2v_dim
        self.bert_dim = bert_dim
        self.fusion_hidden_dim = fusion_hidden_dim

        # ---- 构建融合模块 ----
        self._single_feat = None
        self.fusion = None

        if fusion_type == 'concat':
            self.feat_dim = w2v_dim + bert_dim  # 1068
        elif fusion_type == 'gated_fusion':
            self.fusion = GatedFusion(dim1=w2v_dim, dim2=bert_dim, hidden_dim=fusion_hidden_dim)
            self.feat_dim = fusion_hidden_dim
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(dim1=w2v_dim, dim2=bert_dim, hidden_dim=fusion_hidden_dim)
            self.feat_dim = fusion_hidden_dim
        elif fusion_type == 'proj_fusion':
            self.fusion = ProjFusion(dim1=w2v_dim, dim2=bert_dim, hidden_dim=fusion_hidden_dim)
            self.feat_dim = fusion_hidden_dim
        elif fusion_type == 'w2v_only':
            self._single_feat = 'w2v'
            self.feat_dim = w2v_dim
        elif fusion_type == 'bert_only':
            self._single_feat = 'bert'
            self.feat_dim = bert_dim
        else:
            raise ValueError(f'Unknown fusion type: {fusion_type}')

        # 用于 concat 类型的 FeatureFusion（与 v3 相同）
        self._concat_fusion = None

        # InfoNCE
        self.temprature = temprature
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def set_concat_fusion(self, feat_in_dim, feat_hidden_dim):
        """为 concat 类型设置 FeatureFusion 模块"""
        self._concat_fusion = FeatureFusion(feat_in_dim, feat_hidden_dim)

    def forward_fusion(self, w2v, bert):
        """目标侧特征融合。

        Args:
            w2v:  [B, T, 300] or [N, 300]
            bert: [B, T, 768] or [N, 768]

        Returns:
            fused: [B, T, feat_dim] or [N, feat_dim]
        """
        if self._single_feat == 'w2v':
            return w2v
        if self._single_feat == 'bert':
            return bert

        if self.fusion_type == 'concat':
            return torch.cat([w2v, bert], dim=-1)

        # GatedFusion / CrossAttention / ProjFusion
        if self.fusion_type == 'cross_attention':
            # CrossAttention 中的 MultiheadAttention 会将 T 当作序列长度，
            # 产生 O(T²) 的跨时间注意力。我们希望逐时间步独立融合，
            # 因此将 B*T 折叠为 batch 维，时间步作为 seq_len=1 处理。
            if w2v.dim() == 3:
                B, T, _ = w2v.shape
                w2v_flat = w2v.reshape(B * T, self.w2v_dim)
                bert_flat = bert.reshape(B * T, self.bert_dim)
                fused = self.fusion(w2v_flat, bert_flat)  # [B*T, hidden]
                return fused.reshape(B, T, -1)
            else:
                return self.fusion(w2v, bert)

        # GatedFusion / ProjFusion: Linear + element-wise, 自然兼容 3D 输入
        return self.fusion(w2v, bert)

    def predict(self, x, sub_id):
        return self.brain_encoder(x, sub_id)

    def get_clip_loss_batch(self, x, sub_id, embed):
        """计算 InfoNCE 对比损失。

        Args:
            x:      MEG [B, T, 204]
            sub_id: 被试 ID [B]
            embed:  word2vec+bert 拼接 [B, T, total_dim]

        Returns:
            pred, loss, rank_acc
        """
        # 拆分 embed → word2vec + bert
        if self._single_feat is None:
            w2v = embed[..., :self.w2v_dim]
            bert = embed[..., self.w2v_dim:self.w2v_dim + self.bert_dim]
        elif self._single_feat == 'w2v':
            w2v = embed
            bert = None
        else:
            w2v = None
            bert = embed

        # 目标侧融合
        target = self.forward_fusion(w2v, bert)               # [B, T, feat_dim]

        # concat 类型需要用 FeatureFusion 降维（它期望 [B, D, T] 输入，输出 [B, out, T]）
        if self._concat_fusion is not None:
            target = target.transpose(1, 2).contiguous()      # [B, 1068, T]
            target = self._concat_fusion(target)               # [B, encoder_out, T]
        else:
            target = target.transpose(1, 2).contiguous()      # [B, feat_dim, T]

        # 脑编码器预测
        pred = self.predict(x, sub_id)                         # [B, T, encoder_out_dim]
        pred = pred.transpose(1, 2).contiguous()               # [B, encoder_out_dim, T]

        # Pearson pairwise InfoNCE
        loss, rank_acc = self.info_nce_loss(pred, target)
        return pred, loss, rank_acc

    def info_nce_loss(self, pred, target):
        """Pearson pairwise InfoNCE（与 v3 BrainNetwork 相同）"""
        sim = pearson_pair_wise_batch(pred, target)
        loss = -self.logsoftmax(sim / self.temprature).diag()
        label = torch.arange(sim.shape[0]).to(sim.device)
        _, idx = sim.sort(dim=1, descending=True)
        rank = torch.where(idx == label.unsqueeze(1))[1]
        rank_acc = (sim.shape[1] - 1 - rank) / (sim.shape[1] - 1)
        return loss, rank_acc

    def predict_label(self, x, sub_id, fused_candidates):
        """候选检索（使用预融合的候选嵌入）。

        Args:
            x:                MEG [B, T, 204]
            sub_id:           被试 ID [B]
            fused_candidates: 预融合候选 [N, T, feat_dim]

        Returns:
            top1, top10
        """
        fused_candidates_t = fused_candidates.transpose(1, 2)  # [N, feat_dim, T]

        pred = self.predict(x, sub_id)                         # [B, T, feat_dim]
        pred = pred.transpose(1, 2).contiguous()               # [B, feat_dim, T]

        # 逐帧 Pearson 相关系数取平均（与 v3 相同）
        rr_all = []
        for i in range(pred.shape[1]):
            rr = pearson_pair_wise(pred[:, i, :], fused_candidates_t[:, i, :])
            rr_all.append(rr)
        rr_all = torch.stack(rr_all).mean(dim=0)

        _, top1 = torch.topk(rr_all, k=1)
        _, top10 = torch.topk(rr_all, k=10)
        return top1, top10


# ===========================================================================
# 评估 / 训练
# ===========================================================================

def _prepare_fused_candidates(model, candidate_embeds, device):
    """预计算融合后的候选嵌入。

    Args:
        model:             SegmentFusionNetwork
        candidate_embeds:  [N, T, total_dim] 原始拼接特征
        device:            torch device

    Returns:
        fused: [N, T, feat_dim]
    """
    if model._single_feat is not None:
        # 单特征: 直接取对应部分
        if model._single_feat == 'w2v':
            return candidate_embeds[..., :model.w2v_dim].to(device)
        else:
            return candidate_embeds.to(device)

    if model.fusion_type == 'concat':
        # concat: FeatureFusion 期望 [N, 1068, T] 输入
        w2v = candidate_embeds[..., :model.w2v_dim]
        bert = candidate_embeds[..., model.w2v_dim:model.w2v_dim + model.bert_dim]
        concat = torch.cat([w2v, bert], dim=-1)  # [N, T, 1068]
        if model._concat_fusion is not None:
            concat = concat.transpose(1, 2).contiguous()  # [N, 1068, T]
            fused = model._concat_fusion(concat.to(device))  # [N, out, T]
            return fused.transpose(1, 2).contiguous().detach()  # [N, T, out]
        return concat.to(device)

    # 有可学习参数: 分批处理避免 OOM
    w2v = candidate_embeds[..., :model.w2v_dim]
    bert = candidate_embeds[..., model.w2v_dim:model.w2v_dim + model.bert_dim]

    fused_ls = []
    batch_size = 256
    model.eval()
    with torch.no_grad():
        for i in range(0, len(candidate_embeds), batch_size):
            w2v_b = w2v[i:i + batch_size].to(device)
            bert_b = bert[i:i + batch_size].to(device)
            fused = model.forward_fusion(w2v_b, bert_b)
            fused_ls.append(fused.detach().cpu())
    return torch.cat(fused_ls, dim=0).to(device)


def valid_model(model, dataloaders, device, phase='valid'):
    model.eval()
    if phase == 'valid':
        candidats_idx = torch.tensor(mcfg.feat_valid_id)
    elif phase == "test":
        candidats_idx = torch.tensor(mcfg.feat_test_id)
    else:
        raise ValueError('phase xxx')

    # 预计算融合候选嵌入
    raw_candidates = mcfg.embeds[candidats_idx]
    fused_candidates = _prepare_fused_candidates(model, raw_candidates, device)

    pred_top1 = []
    pred_top10 = []
    target = []
    for iter, (eeg, emb_id, sub_id, _) in enumerate(tqdm(dataloaders[phase]), start=1):
        target_induce = torch.tensor([torch.where(idx == candidats_idx) for idx in emb_id]).squeeze(1)
        eeg, sub_id = eeg.to(device), sub_id.to(device)
        with torch.no_grad():
            pred_top1_induce, pred_top10_induce = model.predict_label(
                x=eeg, sub_id=sub_id, fused_candidates=fused_candidates)
        pred_top1.append(pred_top1_induce.detach().cpu())
        pred_top10.append(pred_top10_induce.detach().cpu())
        target.append(target_induce)

    pred_top1 = torch.cat(pred_top1, dim=0)
    pred_top10 = torch.cat(pred_top10, dim=0)
    target = torch.cat(target, dim=0)
    top1_acc = torch.sum(pred_top1 == target.unsqueeze(1)) / len(target)
    top10_acc = torch.sum(pred_top10 == target.unsqueeze(1), dim=1).sum() / len(target)
    top1_acc = top1_acc.item()
    top10_acc = top10_acc.item()
    return top1_acc, top10_acc, model


def train_model(dataloaders, dataset_sizes, device, model,
                optimizer, scheduler, start_epoch, num_epochs,
                best_top10,
                checkpoint_path_best,
                early_stop_num,
                logger,
                writer):
    since = time.time()

    if best_top10 is None:
        best_top10 = 0
        best_epoch = -1
    else:
        best_epoch = start_epoch - 1

    global_step = 0
    phase = 'train'
    un_update_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()
        running_loss = 0.0
        epoch_rank = None
        for iter, (eeg, emb_id, sub_id, _) in enumerate(tqdm(dataloaders[phase]), start=1):
            embed = mcfg.embeds[emb_id]
            embed = embed.clone().float()
            eeg, embed, sub_id = \
                eeg.to(device), embed.to(device), sub_id.to(device)
            optimizer.zero_grad()
            _, loss, rank = model.get_clip_loss_batch(x=eeg, sub_id=sub_id, embed=embed)
            loss = loss.mean(0, keepdim=True)

            if epoch_rank is None:
                epoch_rank = rank
            else:
                epoch_rank = torch.cat((epoch_rank, rank))

            running_loss += loss.item() * eeg.size()[0]
            loss.backward()
            optimizer.step()
            global_step += 1

        top10_epoch = torch.nonzero(epoch_rank <= 10).shape[0] / epoch_rank.shape[0]
        top5_epoch = torch.nonzero(epoch_rank <= 5).shape[0] / epoch_rank.shape[0]
        top1_epoch = torch.nonzero(epoch_rank == 1).shape[0] / epoch_rank.shape[0]
        sample_size = epoch_rank.shape[0]
        rank_acc_epoch = (sample_size - torch.mean(epoch_rank.float())) / (sample_size - 1)
        writer.add_scalar('Epoch/Train-Loss', running_loss / dataset_sizes[phase], epoch)
        writer.add_scalars('Epoch/Train-Accuracy',
                           {'top-10': top10_epoch,
                            'top-5': top5_epoch,
                            'top-1': top1_epoch,
                            'rank-acc': rank_acc_epoch},
                           epoch)
        logger.info("Train: epoch {:.0f}/ global_step {:.0f}: "
                    "loss: {:.4f}; "
                    "top-10: {:.4f}; "
                    "top-5: {:.4f}; "
                    "top-1: {:.4f}; "
                    "rank-acc: {:.4f}"
                    .format(epoch, global_step,
                            running_loss / dataset_sizes[phase],
                            top10_epoch,
                            top5_epoch,
                            top1_epoch,
                            rank_acc_epoch))
        print('{} Loss: {:.4f}; top-10: {:.4f}; rank-acc: {:.4f}'
              .format(phase, running_loss / dataset_sizes[phase], top10_epoch, rank_acc_epoch))

        print('evaluating valid set...')
        top1_epoch_valid, top10_epoch_valid, model = \
            valid_model(model, dataloaders, device, phase='valid')
        model.train()

        writer.add_scalars(f'Epoch/Valid-Accuracy',
                           {'top-1': top1_epoch_valid,
                            'top-10': top10_epoch_valid},
                           epoch)

        logger.info("Dev: epoch {:.0f}/ global_step {:.0f}: "
                    "top-1: {:.4f}; "
                    "top-10: {:.4f}; "
                    .format(epoch, global_step, top1_epoch_valid, top10_epoch_valid))
        print('valid top-1: {:.4f}, top-10: {:.4f}'
              .format(top1_epoch_valid, top10_epoch_valid))

        if top10_epoch_valid > best_top10:
            best_top10 = top10_epoch_valid
            best_epoch = epoch
            un_update_epoch = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dic": optimizer.state_dict(),
                "epoch": best_epoch,
                "top10_acc": best_top10,
                "global_step": global_step}
            torch.save(checkpoint, checkpoint_path_best)
            print(f'update best on valid checkpoint: {checkpoint_path_best}')
            logger.info(f'update best on valid checkpoint: {checkpoint_path_best}')
        else:
            un_update_epoch += 1
        logger.info("\n\n")

        if un_update_epoch > early_stop_num:
            print(f'early stop at epoch {epoch}, global_step {global_step}')
            logger.info(f'early stop at epoch {epoch}, global_step {global_step}')
            break

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc epoch {:0f}: {:4f}'.format(best_epoch, best_top10))
    logger.info('Best val loss epoch {:0f}: {:4f}'.format(best_epoch, best_top10))

    checkpoint = torch.load(checkpoint_path_best, map_location=device)
    best_model_wts = checkpoint['model_state_dict']
    model.load_state_dict(best_model_wts)

    top1_iter50_test, top10_iter50_test, model = \
        valid_model(model, dataloaders, device, phase='test')
    print('test top-1: {:.4f}, top-10: {:.4f}'.format(top1_iter50_test, top10_iter50_test))
    logger.info('test top-1: {:.4f}, top-10: {:.4f}'.format(top1_iter50_test, top10_iter50_test))

    return model


# ===========================================================================
# 主训练函数
# ===========================================================================

def train(args):
    num_epochs = args.num_epochs
    encoder_name = args.encoder_name
    device = args.device
    batch_size = args.batch_size
    feature_name = args.feature_name
    early_stop_num = args.early_stop_num
    temprature = args.temprature
    out_dim_ls = args.out_dim_ls
    fusion_type = args.fusion_type
    fusion_hidden_dim = args.fusion_hidden_dim
    use_lowpass = args.use_lowpass
    sliding_window = args.sliding_window
    args_dict = vars(args)
    args_json = json.dumps(args_dict, indent=4)

    fs = mcfg.fs
    seg_len = mcfg.seg_len

    bs_tag = f'-bs{batch_size}'
    tp_tag = f'-tp{temprature}'

    # 解析 feature_name → feature_name_ls
    if '_' not in feature_name:
        feature_name_ls = [feature_name]
    else:
        feature_name_ls = feature_name.split('_')
    in_dim_ls = [mcfg.feature_dim_dict[feat] for feat in feature_name_ls]

    # 确定 brain_encoder 输出维度
    if fusion_type in ('gated_fusion', 'cross_attention', 'proj_fusion'):
        out_channels = fusion_hidden_dim
        out_dim_tag = f'-{fusion_hidden_dim}'
    elif fusion_type in ('w2v_only', 'bert_only'):
        out_channels = sum(in_dim_ls)
        out_dim_tag = f'-{out_channels}'
    else:  # concat
        out_channels = sum(out_dim_ls)
        out_dim_ls_str = [str(d) for d in out_dim_ls]
        out_dim_tag = '-'.join(out_dim_ls_str)
        out_dim_tag = f'-{out_dim_tag}'

    model_dir = f'{encoder_name}-{fusion_type}-{feature_name}{bs_tag}{tp_tag}{out_dim_tag}'
    print(f'model dir: {model_dir}')
    checkpoint_dir = op.join(cfg.project_dir, 'result',
                             'segment_fusion_clip_v4', model_dir)
    checkpoint_best_dir = op.join(checkpoint_dir, 'best')
    output_checkpoint_name_best = op.join(checkpoint_best_dir, 'model.pt')
    if not op.exists(checkpoint_best_dir):
        os.makedirs(checkpoint_best_dir)
    tensorboard_dir = op.join(checkpoint_dir, 'runs')
    if not op.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    # Logger
    file_handler = logging.FileHandler(op.join(checkpoint_dir, 'acc_log.txt'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(model_dir)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info('\n\n')
    logger.info('*****************************START NEW SESSION*****************************\n')
    logger.info('PARAMETER ...')
    logger.info(args_json)

    # Random seeds
    seed_val = 1024
    logger.info(f'seed_val: {seed_val}')
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Device
    if not torch.cuda.is_available():
        device = "cpu"
    print(f'[INFO]using device {device}')

    # ---- Build brain encoder ----
    input_dim = 204
    if encoder_name == 'BrainMagic':
        from brain_decoder.brainmagic import BrainMagic
        from model_config import Config_BrainMagic
        config = Config_BrainMagic()
        brain_encoder = BrainMagic(input_channels=input_dim, out_channels=out_channels, sub_num=12)
        lr = config.lr
    elif encoder_name == 'VLAAI':
        config = Config_VLAAI()
        brain_encoder = VLAAI(input_channels=input_dim, out_channels=out_channels, sub_num=12)
        lr = config.lr
    elif encoder_name == 'VLAAI2':
        config = Config_VLAAI()
        brain_encoder = VLAAI2(input_channels=input_dim, out_channels=out_channels, sub_num=12)
        lr = config.lr
    elif encoder_name == 'ConvConCatNet':
        config = Config_ConvConcatNet()
        brain_encoder = ConvConcatNet(input_channels=input_dim, out_channels=out_channels, sub_num=12)
        lr = config.lr
    elif encoder_name == 'WaveNet':
        config = Config_WaveNet()
        brain_encoder = WaveNet(input_channels=input_dim, out_channels=out_channels, sub_num=12)
        lr = config.lr
    elif encoder_name == 'DilatedConvNet':
        config = Config_DilatedConvNet()
        brain_encoder = DilatedConvNet(input_channels=input_dim, out_channels=out_channels, sub_num=12)
        lr = config.lr
    else:
        raise ValueError('encoder_name not supported')

    # ---- Build SegmentFusionNetwork ----
    model = SegmentFusionNetwork(
        brain_encoder=brain_encoder,
        fusion_type=fusion_type,
        w2v_dim=300,
        bert_dim=768,
        fusion_hidden_dim=fusion_hidden_dim,
        temprature=temprature,
    )

    # concat 类型需要 FeatureFusion（与 v3 相同）
    if fusion_type == 'concat':
        model.set_concat_fusion(feat_in_dim=in_dim_ls, feat_hidden_dim=out_dim_ls)

    model.to(device)

    # Tensorboard
    writer = SummaryWriter(tensorboard_dir)

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # ---- Load data ----
    feat_dict_ls = []
    feat_frames_ls = []
    for feat in feature_name_ls:
        if feat == 'bert':
            fd, ffid, fk, ff = get_feat_bert_SEM4L(seg_len=seg_len, fs=fs,
                                                     use_lowpass=use_lowpass,
                                                     sliding_window=sliding_window)
        elif feat == 'word2vec':
            fd, ffid, fk, ff = get_feat_word2vec_SEM4L(seg_len=seg_len, fs=fs,
                                                         use_lowpass=use_lowpass)
        else:
            fd, ffid, fk, ff = get_feat_SEM4L(feature_name=feat, seg_len=seg_len, fs=fs)
        feat_dict_ls.append(fd)
        feat_frames_ls.append(ff)

    feat_dict = {}
    for key in fk:
        feat_dict[key] = np.concatenate([feat_dict_1[key] for feat_dict_1 in feat_dict_ls], axis=-1)
    feat_frames = np.concatenate(feat_frames_ls, axis=-1)
    feat_frames_id_dict = ffid
    feat_keys = fk

    sub_names = ['sub-{:02d}'.format(i) for i in range(1, 13)]

    train_trials = np.arange(1, 46)
    valid_trials = np.arange(46, 51)
    test_trials = np.arange(51, 61)

    print(f'train_trials: {train_trials}')
    print(f'valid_trials: {valid_trials}')
    print(f'test_trials: {test_trials}')
    logger.info(f'train_trials: {train_trials}')
    logger.info(f'valid_trials: {valid_trials}')
    logger.info(f'test_trials: {test_trials}')

    # Train set
    (EEG_dict, EEG_feat_index_dict, Sub_id_dict,
     Stimulus_index_dict, id2dict, train_stimulus_filename) = get_EEG_SEM4L(train_trials,
                                                                               feat_dict,
                                                                               feat_frames_id_dict,
                                                                               feat_keys,
                                                                               fs=fs,
                                                                               seg_len=seg_len)
    train_stimulus_filename = list(set(train_stimulus_filename))
    train_set = SEM4Ldataset(EEG_dict, EEG_feat_index_dict, Sub_id_dict,
                             Stimulus_index_dict,
                             id2dict, sub_names, feat_keys)

    # Valid set
    (EEG_dict, EEG_feat_index_dict, Sub_id_dict,
     Stimulus_index_dict, id2dict, valid_stimulus_filename) = get_EEG_SEM4L(valid_trials,
                                                                               feat_dict,
                                                                               feat_frames_id_dict,
                                                                               feat_keys,
                                                                               fs=fs,
                                                                               seg_len=seg_len)
    valid_set = SEM4Ldataset(EEG_dict, EEG_feat_index_dict, Sub_id_dict,
                             Stimulus_index_dict,
                             id2dict, sub_names, feat_keys)

    # Test set
    (EEG_dict, EEG_feat_index_dict, Sub_id_dict,
     Stimulus_index_dict, id2dict, test_stimulus_filename) = get_EEG_SEM4L(test_trials,
                                                                              feat_dict,
                                                                              feat_frames_id_dict,
                                                                              feat_keys,
                                                                              fs=fs,
                                                                              seg_len=seg_len)
    test_set = SEM4Ldataset(EEG_dict, EEG_feat_index_dict, Sub_id_dict,
                            Stimulus_index_dict,
                            id2dict, sub_names, feat_keys)

    feat_frames_id = [f for f in feat_frames_id_dict.values()]
    embeds = torch.tensor(feat_frames, dtype=torch.float32, device='cpu')

    train_stim_idx = [feat_keys.index(s) for s in train_stimulus_filename]
    train_stim_idx = list(set(train_stim_idx))
    valid_stim_idx = [feat_keys.index(s) for s in valid_stimulus_filename]
    valid_stim_idx = list(set(valid_stim_idx))
    test_stim_idx = [feat_keys.index(s) for s in test_stimulus_filename]
    test_stim_idx = list(set(test_stim_idx))

    feat_train_id = np.concatenate([feat_frames_id[i] for i in train_stim_idx])
    feat_valid_id = np.concatenate([feat_frames_id[i] for i in valid_stim_idx])
    feat_test_id = np.concatenate([feat_frames_id[i] for i in test_stim_idx])

    mcfg.embeds = embeds
    mcfg.feat_frames_id = feat_frames_id
    mcfg.feat_train_id = feat_train_id
    mcfg.feat_valid_id = feat_valid_id
    mcfg.feat_test_id = feat_test_id

    dataset_sizes = {'train': len(train_set), 'test': len(test_set), 'valid': len(valid_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]valid_set size: ', len(valid_set))
    print('[INFO]test_set size: ', len(test_set))

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=16)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16)
    dataloaders = {'train': train_dataloader, 'valid': val_dataloader, 'test': test_dataloader}

    print('=== start training ... ===')
    best_top10 = None
    start_epoch = 0
    logger.info(f'training from the beginning, random initialized')
    print(f'training from the beginning, random initialized')
    train_model(dataloaders, dataset_sizes, device, model,
                optimizer=optimizer, scheduler=exp_lr_scheduler,
                start_epoch=start_epoch, num_epochs=num_epochs,
                best_top10=best_top10,
                checkpoint_path_best=output_checkpoint_name_best,
                early_stop_num=early_stop_num,
                logger=logger,
                writer=writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_name', default=mcfg.encoder_name)
    parser.add_argument('--device', default=mcfg.device)
    parser.add_argument('--num_epochs', type=int, default=mcfg.num_epochs)
    parser.add_argument('--feature_name', default='bert')
    parser.add_argument('--fusion_type', default='concat',
                        choices=['concat', 'gated_fusion', 'cross_attention', 'proj_fusion',
                                 'w2v_only', 'bert_only'],
                        help='融合策略: concat(FeatureFusion), gated_fusion/cross_attention/proj_fusion(模型融合)')
    parser.add_argument('--fusion_hidden_dim', type=int, default=256,
                        help='GatedFusion/CrossAttention/ProjFusion 的隐层维度 (default: 256)')
    parser.add_argument('--early_stop_num', type=int, default=mcfg.early_stop_num)
    parser.add_argument('--batch_size', type=int, default=mcfg.batch_size)
    parser.add_argument('--temprature', type=float, default=mcfg.temprature)
    parser.add_argument('--out_dim_ls', type=int, nargs='+', default=mcfg.out_dim_ls)
    parser.add_argument('--use_lowpass', action='store_true', default=False,
                        help='Enable 4Hz low-pass filter on BERT/word2vec features')
    parser.add_argument('--sliding_window', action='store_true', default=False,
                        help='Use sliding-window BERT embeddings')
    args = parser.parse_args()
    train(args)
