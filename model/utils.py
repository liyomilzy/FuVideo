import os
from datetime import datetime
from typing import Optional, List, Literal

import numpy as np
import torch
import torchvision
import imageio
import cv2
from PIL import Image
from einops import rearrange


def add_watermark(image, watermark_path, wm_rel_size=1/16, boundary=5):
    '''
    Creates a watermark on the saved inference image.
    We request that you do not remove this to properly assign credit to
    Shi-Lab's work.
    '''
    watermark = Image.open(watermark_path)
    w_0, h_0 = watermark.size
    H, W, _ = image.shape
    wmsize = int(max(H, W) * wm_rel_size)
    aspect = h_0 / w_0
    if aspect > 1.0:
        watermark = watermark.resize((wmsize, int(aspect * wmsize)), Image.LANCZOS)
    else:
        watermark = watermark.resize((int(wmsize / aspect), wmsize), Image.LANCZOS)
    w, h = watermark.size
    loc_h = H - h - boundary
    loc_w = W - w - boundary
    image = Image.fromarray(image)
    mask = watermark if watermark.mode in ('RGBA', 'LA') else None
    image.paste(watermark, (loc_w, loc_h), mask)
    return image


def create_video(frames, fps, rescale=False, path=None, watermark=None, exp_name=None, save_frames=False):
    if path is None:
        # 为消融实验创建目录结构
        if exp_name:
            dir = f"experiments/Ablation"
            filename = f"{exp_name}.mp4"
        else:
            dir = "temporal"
            filename = f"movie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, filename)

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)

        if watermark is not None:
            x = add_watermark(x, watermark)
        outputs.append(x)

    ext = os.path.splitext(path)[-1].lower()
    if ext == '.mp4':
        h, w = outputs[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for frame in outputs:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    else:
        imageio.mimsave(path, outputs, fps=fps)

    if save_frames:
        frames_dir = os.path.splitext(path)[0] + '_frames'
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(outputs):
            frame_path = os.path.join(frames_dir, f'frame_{i:04d}.jpg')
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f'[INFO] Saved {len(outputs)} frames to: {frames_dir}')

    return path


class SDXLCausalAttnProcessor:
    """
    专为SDXL设计的因果self-attention处理器
    """
    def __init__(self, unet_chunk_size: int = 2, enabled: bool = True):
        self.unet_chunk_size = int(unet_chunk_size)
        self.enabled = bool(enabled)
        self._source_map = None

    def set_source_map(self, source_map):
        """设置源映射"""
        self._source_map = source_map

    def enable(self, flag: bool = True):
        self.enabled = bool(flag)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 仅对self-attention应用因果映射
        if (not is_cross_attention) and self.enabled and (self._source_map is not None):
            F = key.size(0) // max(1, self.unet_chunk_size)
            if F >= 1:
                key_bfdc = rearrange(key, "(b f) d c -> b f d c", b=self.unet_chunk_size)
                value_bfdc = rearrange(value, "(b f) d c -> b f d c", b=self.unet_chunk_size)

                new_key = key_bfdc.clone()
                new_value = value_bfdc.clone()

                src_map = self._source_map
                if isinstance(src_map, torch.Tensor):
                    src_map = src_map.detach().cpu().tolist()

                for j in range(F):
                    i = int(src_map[j]) if j < len(src_map) else -1
                    if (i >= 0):
                        new_key[:, j] = key_bfdc[:, i]
                        new_value[:, j] = value_bfdc[:, i]

                key = rearrange(new_key, "b f d c -> (b f) d c")
                value = rearrange(new_value, "b f d c -> (b f) d c")

        # 标准attention计算
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SDXLTextConsistencyProcessor:
    """
    修复版本：专为SDXL UNet设计的文本一致性处理器
    """

    def __init__(
        self,
        unet_chunk_size: int = 2,
        consistency_strength: float = 0.8,
        reference_frame_idx: int = 4,
        enabled: bool = True
    ):
        self.unet_chunk_size = unet_chunk_size
        self.consistency_strength = consistency_strength
        self.reference_frame_idx = reference_frame_idx
        self.enabled = enabled

        # 控制参数
        self.active_timesteps = None
        self.active_layers = None
        self.current_timestep = None
        self.current_layer_name = None

    def set_active_timesteps(self, timesteps: Optional[List[int]]):
        """设置激活的时间步"""
        self.active_timesteps = timesteps

    def set_active_layers(self, layer_patterns: Optional[List[str]]):
        """设置激活的层模式"""
        self.active_layers = layer_patterns

    def update_context(self, timestep: int, layer_name: str):
        """更新当前上下文信息"""
        self.current_timestep = timestep
        self.current_layer_name = layer_name

    def should_apply_consistency(self) -> bool:
        """判断是否应该应用一致性处理"""
        if not self.enabled:
            return False

        # 检查时间步
        if self.active_timesteps is not None:
            if self.current_timestep not in self.active_timesteps:
                return False

        # 检查层级
        if self.active_layers is not None:
            if self.current_layer_name is None:
                return False
            if not any(pattern in self.current_layer_name for pattern in self.active_layers):
                return False

        return True

    def apply_frame_consistency_to_attention(
        self,
        attention_probs: torch.Tensor,
        num_frames: int,
        num_heads: int
    ) -> torch.Tensor:
        """
        修复版本：对attention_probs应用帧间一致性

        Args:
            attention_probs: (batch*frames*heads, seq_len_q, seq_len_k) - 3D张量
            num_frames: 帧数
            num_heads: 注意力头数
        """
        if num_frames <= 1 or self.consistency_strength <= 0.0:
            return attention_probs

        batch_heads_frames, seq_len_q, seq_len_k = attention_probs.shape
        batch_size = batch_heads_frames // (num_frames * num_heads)

        # 重排为视频格式: (batch, frames, heads, seq_len_q, seq_len_k)
        try:
            probs_video = attention_probs.view(
                batch_size, num_frames, num_heads, seq_len_q, seq_len_k
            )
        except RuntimeError as e:
            print(f"[DEBUG] Reshape failed: expected {batch_size}x{num_frames}x{num_heads}x{seq_len_q}x{seq_len_k}, got {attention_probs.shape}")
            return attention_probs

        # 获取参考帧的attention pattern
        ref_idx = min(self.reference_frame_idx, num_frames - 1)
        ref_pattern = probs_video[:, ref_idx:ref_idx+1]  # (batch, 1, heads, seq_len_q, seq_len_k)

        # 应用一致性：让所有帧向参考帧靠拢
        consistent_probs = (
            (1 - self.consistency_strength) * probs_video +
            self.consistency_strength * ref_pattern
        )

        # 重排回原格式: (batch*frames*heads, seq_len_q, seq_len_k)
        return consistent_probs.view(batch_heads_frames, seq_len_q, seq_len_k)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        """
        修复版本：主处理函数
        """
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # 计算query（总是来自hidden_states）
        query = attn.to_q(hidden_states)

        # 判断attention类型并设置encoder_hidden_states
        is_cross_attention = encoder_hidden_states is not None
        if not is_cross_attention:
            # self-attention: K,V都来自hidden_states
            encoder_hidden_states = hidden_states
        # cross-attention: K,V来自传入的encoder_hidden_states（文本特征）

        # 计算key和value
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 转换为multi-head格式
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 计算attention scores
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # 关键修复：仅对cross-attention应用一致性处理，并正确处理张量维度
        if is_cross_attention and self.should_apply_consistency():
            num_frames = batch_size // self.unet_chunk_size
            if num_frames > 1:
                try:
                    # 获取注意力头数
                    num_heads = attn.heads

                    attention_probs = self.apply_frame_consistency_to_attention(
                        attention_probs, num_frames, num_heads
                    )
                except Exception as e:
                    print(f"[WARNING] Frame consistency failed: {e}, using original attention")

        # 应用attention到value
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 输出投影
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # Dropout

        return hidden_states


class SelectiveConsistencyProcessor:
    """
    选择性一致性处理器：可精确控制哪些层应用哪种处理
    支持多种文本一致性策略，包括前一帧策略
    """
    def __init__(
        self,
        unet_chunk_size: int = 2,
        # Text consistency参数
        text_consistency_strength: float = 0.8,
        text_consistency_strategy: Literal["reference_frame", "average", "weighted_average", "temporal_smooth", "previous_frame"] = "previous_frame",
        text_reference_frame_idx: int = 0,
        temporal_smooth_weight: float = 0.8,  # 用于temporal_smooth策略
        previous_frame_weight: float = 0.7,   # 新增：用于previous_frame策略
        # Self attention因果对齐参数
        causal_alignment_enabled: bool = True,
        causal_reference_frame_idx: int = 0,
        enabled: bool = True
    ):
        self.unet_chunk_size = unet_chunk_size
        self.enabled = enabled

        # 文本一致性参数
        self.text_consistency_strength = text_consistency_strength
        self.text_consistency_strategy = text_consistency_strategy
        self.text_reference_frame_idx = text_reference_frame_idx
        self.temporal_smooth_weight = temporal_smooth_weight
        self.previous_frame_weight = previous_frame_weight  # 新增

        # 因果对齐参数
        self.causal_alignment_enabled = causal_alignment_enabled
        self.causal_reference_frame_idx = causal_reference_frame_idx

        # 控制参数
        self.active_timesteps = None
        self.current_timestep = None
        self.current_layer_name = None

        # 用于temporal smooth策略的历史记录
        self.attention_history = None
        # 新增：用于previous_frame策略的前一帧记录
        self.previous_frame_attention = None

    def set_active_timesteps(self, timesteps: Optional[List[int]]):
        self.active_timesteps = timesteps

    def update_context(self, timestep: int, layer_name: str):
        self.current_timestep = timestep
        self.current_layer_name = layer_name

    def should_apply_processing(self) -> bool:
        """判断当前时间步是否应该处理"""
        if not self.enabled:
            return False

        if self.active_timesteps is not None:
            if self.current_timestep not in self.active_timesteps:
                return False

        return True

    def apply_text_consistency(self, attention_probs, num_frames, num_heads):
        """应用文本一致性（仅cross-attention）- 支持多种策略包括前一帧"""
        if num_frames <= 1 or self.text_consistency_strength <= 0.0:
            return attention_probs

        batch_heads_frames, seq_len_q, seq_len_k = attention_probs.shape
        batch_size = batch_heads_frames // (num_frames * num_heads)

        try:
            probs_video = attention_probs.view(
                batch_size, num_frames, num_heads, seq_len_q, seq_len_k
            )

            if self.text_consistency_strategy == "reference_frame":
                # 原始策略：对齐到参考帧
                ref_idx = min(self.text_reference_frame_idx, num_frames - 1)
                target_pattern = probs_video[:, ref_idx:ref_idx+1]

            elif self.text_consistency_strategy == "average":
                # 策略：所有帧的平均
                target_pattern = probs_video.mean(dim=1, keepdim=True)

            elif self.text_consistency_strategy == "weighted_average":
                # 加权平均：中心帧权重更高
                weights = torch.zeros(num_frames, device=attention_probs.device)
                center_idx = num_frames // 2
                for i in range(num_frames):
                    # 距离中心越近权重越高
                    distance = abs(i - center_idx)
                    weights[i] = torch.exp(-distance * 0.5)
                weights = weights / weights.sum()

                # 应用权重
                weighted_probs = probs_video * weights.view(1, -1, 1, 1, 1)
                target_pattern = weighted_probs.sum(dim=1, keepdim=True)

            elif self.text_consistency_strategy == "temporal_smooth":
                # 时序平滑：当前平均 + 历史记录
                current_avg = probs_video.mean(dim=1, keepdim=True)

                if self.attention_history is None:
                    target_pattern = current_avg
                    self.attention_history = current_avg.clone()
                else:
                    # 指数移动平均
                    self.attention_history = (
                        (1 - self.temporal_smooth_weight) * self.attention_history +
                        self.temporal_smooth_weight * current_avg
                    )
                    target_pattern = self.attention_history

            elif self.text_consistency_strategy == "previous_frame":
                # 新增：前一帧策略
                target_pattern = self._apply_previous_frame_strategy(probs_video, num_frames)

            else:
                raise ValueError(f"Unknown text consistency strategy: {self.text_consistency_strategy}")

            # 应用一致性
            consistent_probs = (
                (1 - self.text_consistency_strength) * probs_video +
                self.text_consistency_strength * target_pattern
            )

            return consistent_probs.view(batch_heads_frames, seq_len_q, seq_len_k)
        except RuntimeError as e:
            print(f"[WARN] Text consistency failed: {e}")
            return attention_probs

    def _apply_previous_frame_strategy(self, probs_video, num_frames):
        """
        新增：前一帧策略的具体实现

        核心思想：每一帧都参考其前一帧的注意力模式，建立时序依赖
        """
        batch_size, num_frames, num_heads, seq_len_q, seq_len_k = probs_video.shape

        # 处理第一帧：没有前一帧，使用自身
        if self.previous_frame_attention is None or self.previous_frame_attention.shape[1] != num_frames:
            # 初始化或帧数不匹配时重置
            target_pattern = probs_video[:, :1].clone()  # 第一帧用自身
        else:
            # 使用前一次的注意力模式作为目标
            target_pattern = self.previous_frame_attention.clone()

        # 为每一帧构建目标模式
        frame_targets = []

        for frame_idx in range(num_frames):
            if frame_idx == 0:
                # 第一帧：使用前一时间步的第一帧或自身
                if self.previous_frame_attention is not None:
                    frame_target = self.previous_frame_attention[:, 0:1]
                else:
                    frame_target = probs_video[:, 0:1]  # 初始情况使用自身
            else:
                # 其他帧：使用当前时间步的前一帧
                # 这里使用当前batch内的前一帧，建立帧间依赖
                frame_target = probs_video[:, frame_idx-1:frame_idx]

            frame_targets.append(frame_target)

        # 拼接所有帧的目标模式
        target_pattern = torch.cat(frame_targets, dim=1)

        # 更新历史记录为当前的注意力模式（用于下一个时间步）
        self.previous_frame_attention = probs_video.clone().detach()

        return target_pattern

    def apply_causal_alignment(self, key, value, num_frames):
        """应用因果对齐（仅self-attention）- 简化的前一帧策略"""
        if not self.causal_alignment_enabled or num_frames <= 1:
            return key, value

        try:
            key_video = rearrange(key, "(b f) d c -> b f d c", b=self.unet_chunk_size)
            value_video = rearrange(value, "(b f) d c -> b f d c", b=self.unet_chunk_size)

            # 建立前一帧依赖链：1->2->3->4...
            aligned_key_list = []
            aligned_value_list = []

            for frame_idx in range(num_frames):
                if frame_idx == 0:
                    # 第一帧使用自身
                    target_key = key_video[:, 0:1]
                    target_value = value_video[:, 0:1]
                else:
                    # 其他帧使用前一帧的key/value
                    target_key = key_video[:, 0:1]
                    target_value = value_video[:, 0:1]

                aligned_key_list.append(target_key)
                aligned_value_list.append(target_value)

            # 拼接所有帧
            aligned_key_video = torch.cat(aligned_key_list, dim=1)
            aligned_value_video = torch.cat(aligned_value_list, dim=1)

            aligned_key = rearrange(aligned_key_video, "b f d c -> (b f) d c")
            aligned_value = rearrange(aligned_value_video, "b f d c -> (b f) d c")

            return aligned_key, aligned_value

        except Exception as e:
            print(f"[WARN] Causal alignment failed: {e}")
            return key, value

    def reset_history(self):
        """重置历史记录（用于新的生成任务）"""
        self.attention_history = None
        self.previous_frame_attention = None  # 新增：重置前一帧记录

    def get_strategy_info(self):
        """获取当前策略信息"""
        info = {
            "text_consistency_strategy": self.text_consistency_strategy,
            "text_consistency_strength": self.text_consistency_strength,
            "causal_alignment_enabled": self.causal_alignment_enabled,
            "enabled": self.enabled
        }

        if self.text_consistency_strategy == "reference_frame":
            info["reference_frame_idx"] = self.text_reference_frame_idx
        elif self.text_consistency_strategy == "temporal_smooth":
            info["temporal_smooth_weight"] = self.temporal_smooth_weight
        elif self.text_consistency_strategy == "previous_frame":  # 新增
            info["previous_frame_weight"] = self.previous_frame_weight

        return info

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if not is_cross_attention:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 根据attention类型和配置决定是否处理
        if self.should_apply_processing():
            num_frames = batch_size // self.unet_chunk_size

            if not is_cross_attention and self.causal_alignment_enabled:
                # Self-attention: 应用因果对齐
                key, value = self.apply_causal_alignment(key, value, num_frames)

        # 标准attention计算
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Cross-attention的文本一致性处理
        if is_cross_attention and self.should_apply_processing():
            num_frames = batch_size // self.unet_chunk_size
            if num_frames > 1:
                num_heads = attn.heads
                attention_probs = self.apply_text_consistency(
                    attention_probs, num_frames, num_heads
                )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
