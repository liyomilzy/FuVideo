import logging
import torch
import math

from typing import Union, List, Optional, Callable, Tuple, Dict, Any
from dataclasses import dataclass
from typing import Optional, List, Literal
from diffusers import DiffusionPipeline, DDPMScheduler
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from einops import rearrange
import PIL
import numpy as np

# 导入SDXL兼容的处理器 - 修改这里支持ProgressiveConsistencyProcessor
from .utils import SDXLTextConsistencyProcessor,SDXLCausalAttnProcessor,SelectiveConsistencyProcessor
# 新增：导入序列对齐工具

try:
    from diffusers.utils import logging as diffusers_logging
    logger = diffusers_logging.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class FuVideoPipelineOutput(BaseOutput):
    """视频生成管道的输出"""
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


class FuVideoPipeline(DiffusionPipeline, FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin):
    r"""
    基于 Stable Diffusion XL 的文本到视频生成管道 - DDIM→DDPM混合采样版本
    
    **新增功能**：渐变提示词支持 + Embedding插值，实现动作/状态的平滑过渡
    
    修改后的采样策略：
    1. DDPM采样：从全噪声快速到半噪声状态，建立整体结构和语义（带embedding插值）
    2. DDIM采样：从半噪声精细化到最终结果，增强细节质量（带embedding插值+注意力策略）
    """
    
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    _optional_components = ["image_encoder", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size
        self._original_processors = {}
        self._causal_processors = {}            # self-attention processors
        self._text_consistency_processors = {}  # cross-attention processors
        # 初始化水印（如果需要）
        add_watermarker = add_watermarker if add_watermarker is not None else False
        if add_watermarker:
            try:
                from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker
                self.watermark = StableDiffusionXLWatermarker()
            except ImportError:
                logger.warning("Watermarker not available, skipping watermark")
                self.watermark = None
        else:
            self.watermark = None

    def enable_vae_slicing(self):
        """Enable sliced VAE decoding"""
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        """Disable sliced VAE decoding"""
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        """Enable tiled VAE decoding"""
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        """Disable tiled VAE decoding"""
        self.vae.disable_tiling()

    def _calculate_switch_timestep(self, ddim_ratio: float, total_train_timesteps: int = 1000) -> int:
        """计算DDIM到DDPM的切换时间步"""
        t_switch = int(total_train_timesteps * (1 - ddim_ratio))
        print(f"[INFO] Calculated switch timestep: {t_switch} (DDPM handles the first {ddim_ratio*100:.1f}% of the schedule)")
        return t_switch

    # ========================================================================
    # 新增：Embedding插值核心方法
    # ========================================================================
    
    def _subset_semantic_points(
        self,
        semantic_points,
        frame_indices: list[int],
    ) -> list[float] | None:
        """按输出帧索引从完整 semantic_points 中取出子集（如 DDPM 仅处理奇数帧时）。"""
        if semantic_points is None:
            return None
        s = [float(x) for x in semantic_points]
        total = len(s)
        for idx in frame_indices:
            if idx < 0 or idx >= total:
                raise ValueError(
                    f"frame index {idx} out of range for semantic_points (length {total})"
                )
        return [s[i] for i in frame_indices]

    def _align_semantic_points(
        self,
        semantic_points,
        video_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        将 semantic_points 与当前批次的 video_length 一一对应（不做重采样）。

        完整视频有 N 帧时，用户传入 N 个 s_i；若本批次只处理部分帧（如奇数帧），
        调用方应先用 _subset_semantic_points 按帧索引取出子集再传入。
        """
        s_np = np.asarray(semantic_points, dtype=np.float32).reshape(-1)
        n = s_np.shape[0]
        if n != video_length:
            raise ValueError(
                f"semantic_points length ({n}) must equal video_length ({video_length}) "
                f"for this interpolation batch. "
                f"For partial-frame passes, pass the indexed subset of semantic_points."
            )
        return torch.from_numpy(s_np).to(device=device, dtype=torch.float32)

    def interpolate_embeddings_for_video(
        self,
        start_embeds: torch.FloatTensor,
        end_embeds: torch.FloatTensor,
        video_length: int,
        interpolation_strategy: str = "linear",
        device: torch.device = None,
        semantic_points=None,
        power_fast_exponent: float = 0.4,
    ) -> torch.FloatTensor:
        """
        为视频的每一帧插值计算prompt embeddings
        
        Args:
            start_embeds: 起始embedding [batch_size, seq_len, hidden_size]
            end_embeds: 结束embedding [batch_size, seq_len, hidden_size]  
            video_length: 视频帧数
            interpolation_strategy: 插值策略（当 semantic_points 为 None 时生效）
            semantic_points: LLM 给出的归一化语义时间点列表 s（s_0=0, s_{n-1}=1），
                             长度必须等于 video_length（逐帧一一对应，不重采样）。
                             一旦提供，将忽略 interpolation_strategy，改用
                             w_i = s_i ^ power_fast_exponent 作为插值权重。
            power_fast_exponent: power_fast 映射指数，默认 0.4。
            
        Returns:
            torch.FloatTensor: [batch_size * video_length, seq_len, hidden_size]
        """
        if device is None:
            device = start_embeds.device
            
        # 生成插值权重
        if video_length == 1:
            weights = torch.tensor([0.0], device=device)
        elif semantic_points is not None:
            # ===== LLM 语义时间点（逐帧一一对应）+ power_fast 偏置矫正 =====
            s_values = self._align_semantic_points(semantic_points, video_length, device)
            weights = torch.pow(s_values.clamp(0.0, 1.0), power_fast_exponent)
            print(f"[INFO] Using LLM semantic points (n={len(semantic_points)}, 1:1 per frame) -> "
                  f"power_fast(exp={power_fast_exponent}) for {video_length} frames")
        else:
            t = torch.linspace(0, 1, video_length, device=device)
            
            if interpolation_strategy == "linear":
                weights = t
            elif interpolation_strategy == "cosine":
                # sin(πt/2): 快速开头，缓慢收尾
                weights = torch.sin(math.pi * t / 2)
            elif interpolation_strategy == "ease_in_out":
                # t²(3-2t): S形曲线，两端慢中间快
                weights = t * t * (3 - 2 * t)
            elif interpolation_strategy == "power_fast":
                # t^0.4: 极速开头，frame1 即达到 ~50% end 语义
                # 抵消 self-attention 对 frame0 结构的锚定偏移
                weights = torch.pow(t, 0.4)
            elif interpolation_strategy == "offset_cosine":
                # 0.15 + 0.85*sin(πt/2): 预热偏移 + 余弦曲线
                # frame0 已包含 15% end 语义，补偿 self-attn 第一帧锚定
                offset = 0.15
                weights = offset + (1.0 - offset) * torch.sin(math.pi * t / 2)
            else:
                weights = t  # 默认线性
        
        # 为每帧计算插值embedding
        interpolated_frames = []
        
        for frame_idx in range(video_length):
            weight = weights[frame_idx].item()
            
            # 线性插值: (1-w)*start + w*end
            frame_embedding = (1.0 - weight) * start_embeds + weight * end_embeds
            interpolated_frames.append(frame_embedding)
        
        # 拼接所有帧: [video_length * batch_size, seq_len, hidden_size]
        result = torch.cat(interpolated_frames, dim=0)
        
        print(f"[DEBUG] Interpolated embeddings: {len(interpolated_frames)} frames, strategy={interpolation_strategy}")
        print(f"[DEBUG] Weight range: {weights[0]:.3f} -> {weights[-1]:.3f}")
        
        return result

    # ========================================================================
    # 原有方法保持不变
    # ========================================================================

    def setup_selective_consistency_attention(
        self,
        unet_chunk_size: int = 2,
        # Text consistency控制
        text_layers: Optional[List[str]] = None,  # 指定哪些层应用文本一致性
        text_consistency_strength: float = 0.8,
        # Self attention因果对齐控制
        causal_layers: Optional[List[str]] = None,  # 指定哪些层应用因果对齐
        causal_reference_frame_idx: int = 0,
        # 时间步控制
        active_timesteps: Optional[List[int]] = None,
    ):
        """设置选择性一致性注意力：精确控制每层的处理方式"""
        
        # 默认层配置
        if text_layers is None:
            text_layers = [
                'down_blocks.1',
                'down_blocks.2', 
                'mid_block',
                'up_blocks.0',
                'up_blocks.1'
            ]
        
        if causal_layers is None:
            causal_layers = [
                
                'mid_block',
                'up_blocks.1'
                
            ]
        
        for name, module in self.unet.named_modules():
            if hasattr(module, "processor") and ("attn1" in name or "attn2" in name):
                if name not in self._original_processors:
                    self._original_processors[name] = module.processor
                
                # 判断当前层应该应用哪种处理
                should_apply_text = "attn2" in name and any(pattern in name for pattern in text_layers)
                should_apply_causal = "attn1" in name and any(pattern in name for pattern in causal_layers)
                
                if should_apply_text or should_apply_causal:
                    processor = SelectiveConsistencyProcessor(
                        unet_chunk_size=unet_chunk_size,
                        text_consistency_strength=text_consistency_strength if should_apply_text else 0.0,
                        causal_alignment_enabled=should_apply_causal,
                        causal_reference_frame_idx=causal_reference_frame_idx,
                        enabled=True
                    )
                    
                    if active_timesteps:
                        processor.set_active_timesteps(active_timesteps)
                        
                    self._text_consistency_processors[name] = processor
                    module.set_processor(processor)
                    
                    print(f"[INFO] Layer {name}: text_consistency={should_apply_text}, causal_alignment={should_apply_causal}")
        
        print(f"[INFO] Set up selective consistency for {len(self._text_consistency_processors)} layers")

    def setup_causal_attention(self, unet_chunk_size: int = 2):
        """设置因果注意力机制 - SDXL兼容版本"""
        
        # 保存原始处理器
        for name, module in self.unet.named_modules():
            if hasattr(module, "processor") and "attn1" in name:  # self-attention层
                if name not in self._original_processors:
                    self._original_processors[name] = module.processor
                    
                # 设置SDXL兼容的因果注意力处理器
                causal_proc = SDXLCausalAttnProcessor(unet_chunk_size=unet_chunk_size)
                self._causal_processors[name] = causal_proc
                module.set_processor(causal_proc)
        
        print(f"[INFO] Set up SDXL causal attention for {len(self._causal_processors)} layers")

    def update_attention_context(self, timestep: int):
        """更新所有处理器的上下文信息"""
        for name, processor in self._text_consistency_processors.items():
            if hasattr(processor, 'update_context'):
                processor.update_context(timestep, name)
            
    def set_text_consistency_strength(self, strength: float):
        """动态调整一致性强度"""
        for processor in self._text_consistency_processors.values():
            processor.consistency_strength = strength
            
    def enable_text_consistency(self, enabled: bool = True):
        """启用/禁用文本一致性处理"""
        for processor in self._text_consistency_processors.values():
            processor.enabled = enabled

    def restore_original_attention(self):
        """恢复原始注意力机制"""
        for name, module in self.unet.named_modules():
            if name in self._original_processors:
                module.set_processor(self._original_processors[name])
        print("[INFO] Restored original attention processors")
 

    def encode_simple_progressive_prompts(
        self,
        start_prompt: str,
        end_prompt: str,
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """简化的渐变提示词编码：只返回 start 和 end 的嵌入"""
        device = device or self._execution_device
        
        print(f"[INFO] Encoding progressive prompts:")
        print(f"  Start: '{start_prompt}'")
        print(f"  End: '{end_prompt}'")
        
        # 分别编码两个提示词
        start_embeds, start_neg_embeds, start_pooled, start_neg_pooled = self.encode_prompt(
            prompt=start_prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        
        end_embeds, end_neg_embeds, end_pooled, end_neg_pooled = self.encode_prompt(
            prompt=end_prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        
        # 如果启用 CFG，合并正负嵌入
        if do_classifier_free_guidance:
            start_combined = torch.cat([start_neg_embeds, start_embeds], dim=0)
            end_combined = torch.cat([end_neg_embeds, end_embeds], dim=0)
        else:
            start_combined = start_embeds
            end_combined = end_embeds
            
        return start_combined, end_combined


    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        """编码提示词 - 与父类完全一致"""
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt, negative_prompt_2]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    ):
        """输入验证 - 与父类一致"""
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )



    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        """获取SDXL额外时间ID - 与父类一致"""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def upcast_vae(self):
        """VAE精度提升 - 与父类一致"""
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)

    def decode_video_latents(self, latents):
        """将潜在变量解码为视频帧 - 完全遵循父类VAE处理逻辑"""
        print(f"[DEBUG] Decoding video latents: {latents.shape}")
        print(f"[DEBUG] Latents range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
        
        video_length = latents.shape[2]
        
        # 重排为 (batch*frames, channels, height, width)
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        
        # === 完全遵循父类VAE处理逻辑 ===
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        
        # 重排回视频格式
        image = rearrange(image, "(b f) c h w -> b c f h w", f=video_length)
        image = (image/ 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.detach().cpu()
        print(f"[DEBUG] Decoded video range: [{image.min().item():.4f}, {image.max().item():.4f}]")
        
        return image

    # ========================================================================
    # 修改后的DDPM和DDIM方法 - 添加embedding插值支持
    # ========================================================================
    
    
    def _ddpm_structure_building_with_interpolation(
            self,
            single_frame_latents: torch.FloatTensor,
            video_length: int,
            ddpm_steps: int,     # 保留但此版本为“最准确”：内部强制 ratio=1 精确走到 t_start
            t_start: int,
            # 新增：插值参数
            start_embeds: torch.FloatTensor,
            end_embeds: torch.FloatTensor,

            # 原有参数
            add_text_embeds: torch.FloatTensor,
            add_time_ids: torch.FloatTensor,
            do_classifier_free_guidance: bool,
            guidance_scale: float,
            interpolation_strategy: str = "linear",
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            text_consistency_strength: float = 1,
            text_consistency_active_layers: Optional[List[str]] = None,
            causal_layers: Optional[List[str]] = None,
            causal_reference_frame_idx: int = 0,
            attention_strategy: str = "first_frame_broadcast",
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            semantic_points=None,
            power_fast_exponent: float = 0.4,
        ) -> torch.FloatTensor:
        """DDPM结构建立阶段：只处理奇数帧，完成后在 t=t_start 处做严格噪声空间插值到 2n-1 帧（含分布投影）"""
        print(f"[INFO] Starting DDPM structure building - odd frames only, then strict interpolation to even frames")
        print(f"[INFO] Target: {video_length} frames -> DDPM on {(video_length+1)//2} odd frames -> interpolate to {2*((video_length+1)//2)-1} frames")

        batch_size, c, h, w = single_frame_latents.shape
        device = single_frame_latents.device
        
        # -----------------------------------------------------------------------------
        # 1) 构建 DDPM scheduler，并强制 ratio=1 以精确走到 t_start
        # -----------------------------------------------------------------------------
        try:
            ddpm_scheduler = DDPMScheduler.from_config(self.scheduler.config)
        except Exception as e:
            print(f"[WARNING] Could not create DDPM scheduler from config: {e}")
            ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000)

        num_train_timesteps = ddpm_scheduler.config.num_train_timesteps
        ddpm_scheduler.set_timesteps(num_train_timesteps, device=device)  # ratio=1, timesteps=[999,998,...,0]
        full_ts = ddpm_scheduler.timesteps.tolist()                   # 递减列表

        # 只跑到 t_start：最后一次 step 用 t = t_start+1，这样 prev_sample 就在 t_start
        start_timestep = 999  # 从最大噪声开始
        ddpm_timesteps = list(range(start_timestep, t_start - 1, -(start_timestep - t_start) // ddpm_steps))
        if len(ddpm_timesteps) != ddpm_steps:
            ddpm_timesteps = ddpm_timesteps[:ddpm_steps]
        

        t_eff = t_start  # 最终半噪声所在时间步

        # -----------------------------------------------------------------------------
        # 2) 奇偶帧索引
        # -----------------------------------------------------------------------------
        odd_frame_indices = list(range(0, video_length, 2))
        even_frame_indices = list(range(1, video_length, 2))
        num_odd_frames = len(odd_frame_indices)
        print(f"[INFO] Processing odd frames (1-based): {[i+1 for i in odd_frame_indices]} ({num_odd_frames} frames)")
        print(f"[INFO] Will interpolate even frames (1-based): {[i+1 for i in even_frame_indices]} after DDPM")
        if video_length > 0:
            print(f"[INFO] Expected computation reduction vs full denoise: {len(even_frame_indices)/video_length*100:.1f}%")

        # -----------------------------------------------------------------------------
        # 3) 选择性注意力（与你原逻辑一致）
        # -----------------------------------------------------------------------------
        active_timesteps = ddpm_timesteps
        self.setup_selective_consistency_attention(
            unet_chunk_size=2 if do_classifier_free_guidance else 1,
            text_layers=text_consistency_active_layers ,
            text_consistency_strength=text_consistency_strength,
            causal_layers=causal_layers ,
            causal_reference_frame_idx=causal_reference_frame_idx,
            active_timesteps=active_timesteps,
        )


        # -----------------------------------------------------------------------------
        # 4) 仅为奇数帧生成 prompt embeddings（CFG 支持）
        # -----------------------------------------------------------------------------
        odd_semantic_points = self._subset_semantic_points(semantic_points, odd_frame_indices)

        if do_classifier_free_guidance:
            expected_cfg_size = start_embeds.shape[0] // 2
            if expected_cfg_size == 0:
                raise ValueError("CFG enabled but embeddings don't contain CFG structure")

            start_neg, start_pos = start_embeds[:expected_cfg_size], start_embeds[expected_cfg_size:]
            end_neg, end_pos = end_embeds[:expected_cfg_size], end_embeds[expected_cfg_size:]

            interpolated_pos = self.interpolate_embeddings_for_video(
                start_embeds=start_pos, end_embeds=end_pos,
                video_length=num_odd_frames, interpolation_strategy=interpolation_strategy, device=device,
                semantic_points=odd_semantic_points, power_fast_exponent=power_fast_exponent
            )
            interpolated_neg = self.interpolate_embeddings_for_video(
                start_embeds=start_neg, end_embeds=end_neg,
                video_length=num_odd_frames, interpolation_strategy=interpolation_strategy, device=device,
                semantic_points=odd_semantic_points, power_fast_exponent=power_fast_exponent
            )
            interpolated_prompt_embeds = torch.cat([interpolated_neg, interpolated_pos], dim=0)
        else:
            interpolated_prompt_embeds = self.interpolate_embeddings_for_video(
                start_embeds=start_embeds, end_embeds=end_embeds,
                video_length=num_odd_frames, interpolation_strategy=interpolation_strategy, device=device,
                semantic_points=odd_semantic_points, power_fast_exponent=power_fast_exponent
            )

        # -----------------------------------------------------------------------------
        # 5) 初始化奇数帧 latents
        # -----------------------------------------------------------------------------
        odd_latents_list = [single_frame_latents.clone() for _ in range(num_odd_frames)]
        current_latents = torch.cat(odd_latents_list, dim=0)  # [B*num_odd, C, H, W]
        print(f"[DEBUG] Initialized odd frames latents: {current_latents.shape}")

        # 扩展附加条件到奇数帧
        if do_classifier_free_guidance:
            cfg_chunk_size = add_text_embeds.shape[0] // 2
            neg_text_embeds, pos_text_embeds = add_text_embeds[:cfg_chunk_size], add_text_embeds[cfg_chunk_size:]
            neg_time_ids,  pos_time_ids  = add_time_ids[:cfg_chunk_size],  add_time_ids[cfg_chunk_size:]

            expanded_neg_text = neg_text_embeds.repeat(num_odd_frames, 1)
            expanded_pos_text = pos_text_embeds.repeat(num_odd_frames, 1)
            expanded_neg_time = neg_time_ids.repeat(num_odd_frames, 1)
            expanded_pos_time = pos_time_ids.repeat(num_odd_frames, 1)

            expanded_add_text_embeds = torch.cat([expanded_neg_text, expanded_pos_text], dim=0)
            expanded_add_time_ids    = torch.cat([expanded_neg_time,  expanded_pos_time], dim=0)
        else:
            expanded_add_text_embeds = add_text_embeds.repeat(num_odd_frames, 1)
            expanded_add_time_ids    = add_time_ids.repeat(num_odd_frames, 1)

        # -----------------------------------------------------------------------------
        # 6) DDPM 反演，仅在奇数帧上，从 999 连续到 t_start
        # -----------------------------------------------------------------------------
        print(f"[INFO] Running DDPM on {num_odd_frames} odd frames only (exact to t={t_start})")

        with self.progress_bar(total=len(ddpm_timesteps)) as progress_bar:
            for step_idx, t in enumerate(ddpm_timesteps):
                self.update_attention_context(t)

                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([current_latents] * 2, dim=0)
                    prompt_embeds_input = interpolated_prompt_embeds
                    text_embeds_input   = expanded_add_text_embeds
                    time_ids_input      = expanded_add_time_ids
                else:
                    latent_model_input = current_latents
                    prompt_embeds_input = interpolated_prompt_embeds
                    text_embeds_input   = expanded_add_text_embeds
                    time_ids_input      = expanded_add_time_ids

                if step_idx == 0:
                    print(f"[DEBUG] DDPM Step {step_idx} - odd frames only:")
                    print(f"  latents: {current_latents.shape}")
                    print(f"  model_input: {latent_model_input.shape}")
                    print(f"  prompt_embeds: {prompt_embeds_input.shape}")

                latent_model_input = ddpm_scheduler.scale_model_input(latent_model_input, t)
                t_tensor = torch.tensor(t, device=device, dtype=torch.long)

                with torch.no_grad():
                    added_cond_kwargs = {"text_embeds": text_embeds_input, "time_ids": time_ids_input}
                    noise_pred = self.unet(
                        latent_model_input, t_tensor,
                        encoder_hidden_states=prompt_embeds_input,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # ratio=1，step(t) -> prev_sample at t-1；当 t==t_start+1 时得到 t_start
                current_latents = ddpm_scheduler.step(
                    noise_pred, t_tensor, current_latents, generator=generator
                ).prev_sample

                progress_bar.update()
                if callback is not None and step_idx % callback_steps == 0:
                    callback_latents = rearrange(current_latents, "(f b) c h w -> b c f h w", f=num_odd_frames, b=batch_size)
                    callback(step_idx, t, callback_latents)

        # 奇数帧半噪声（t=t_start）
        odd_half_noise = rearrange(current_latents, "(f b) c h w -> b c f h w", f=num_odd_frames, b=batch_size)
        print(f"[INFO] DDPM completed at t={t_eff} - odd frames half-noise: {odd_half_noise.shape}")

        # -----------------------------------------------------------------------------
        # 7) 严格噪声空间插值（UNet + 投影回 q_t）
        # -----------------------------------------------------------------------------
        print(f"[INFO] Applying strict noise-space interpolation with projection to q_t")
        interpolated_result = self._apply_strict_noise_space_interpolation(
            half_noise_latents=odd_half_noise,
            ddpm_scheduler=ddpm_scheduler,
            t_eff=t_eff,
            start_embeds=start_embeds,
            end_embeds=end_embeds,
            add_text_embeds=add_text_embeds,
            add_time_ids=add_time_ids,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=guidance_scale,
            interpolation_strategy=interpolation_strategy,
            original_video_length=video_length,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
            project_back_to_qt=True,   # 强烈建议开启
            semantic_points=semantic_points,
            power_fast_exponent=power_fast_exponent,
        )

        print(f"[INFO] Final result: {odd_half_noise.shape} -> {interpolated_result.shape}")
        return interpolated_result


    def _apply_strict_noise_space_interpolation(
        self,
        half_noise_latents: torch.FloatTensor,  # [B, C, F_odd, H, W] at t=t_eff
        ddpm_scheduler,
        t_eff: int,                             # 实际处于的时间步（等于 t_start）
        start_embeds: torch.FloatTensor,
        end_embeds: torch.FloatTensor,
        add_text_embeds: torch.FloatTensor,
        add_time_ids: torch.FloatTensor,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        interpolation_strategy: str,
        original_video_length: int,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        project_back_to_qt: bool = True,
        semantic_points=None,
        power_fast_exponent: float = 0.4,
    ) -> torch.FloatTensor:
        """
        严格噪声空间插值（UNet 参与）：
        1) 预测相邻奇数帧的 ε 与 ˆx0；
        2) 对 ε 做相关性归一化插值（lam=0.5），x0 做线性插值；
        3) 合成偶数帧 x_t；
        4) （可选）做一轮“去噪→再加噪”投影回 q_t 以纠正分布偏差。
        """
        B, C, F_odd, H, W = half_noise_latents.shape
        device = half_noise_latents.device
        if F_odd <= 1:
            return half_noise_latents

        target_frames = 2 * F_odd - 1
        print(f"[INFO] Noise space interpolation: {F_odd} -> {target_frames} frames @ t={t_eff}")

        # 生成完整视频长度（target_frames）的条件，用于偶数帧噪声预测/投影
        full_embeddings = self._generate_full_embeddings_for_interpolation(
            start_embeds, end_embeds, add_text_embeds, add_time_ids,
            video_length=target_frames,
            do_classifier_free_guidance=do_classifier_free_guidance,
            interpolation_strategy=interpolation_strategy,
            device=device,
            semantic_points=semantic_points,
            power_fast_exponent=power_fast_exponent,
        )

        # 扩散参数
        alpha_bar_t = ddpm_scheduler.alphas_cumprod[t_eff].to(device)     # 𝛼̄_t
        sqrt_ab     = torch.sqrt(alpha_bar_t).to(device)
        sqrt_omb    = torch.sqrt(1.0 - alpha_bar_t).to(device)

        # 用于可选投影：一小步 t->t-1 的 α_t
        if t_eff > 0:
            alpha_bar_prev = ddpm_scheduler.alphas_cumprod[t_eff - 1].to(device)
            alpha_step = (alpha_bar_t / alpha_bar_prev).clamp(1e-6, 1.0)  # α_t
            sqrt_alpha_step = torch.sqrt(alpha_step).to(device)
            sqrt_one_minus_alpha_step = torch.sqrt(1.0 - alpha_step).to(device)

        frames_out = []
        lam = 0.5  # 单个偶数帧时自然权重

        def odd_target_index(k: int) -> int:
            return 2 * k

        for k in range(F_odd - 1):
            # 追加左侧奇数帧
            frames_out.append(half_noise_latents[:, :, k:k+1])

            # 取相邻奇数帧的 x_t (t=t_eff)
            xL = half_noise_latents[:, :, k, :, :]     # [B,C,H,W]
            xR = half_noise_latents[:, :, k+1, :, :]

            # 预测 ε 并恢复 ˆx0
            eps_L = self._predict_noise_single_frame(
                xL, t_eff, odd_target_index(k),
                full_embeddings,
                do_classifier_free_guidance, guidance_scale,
                cross_attention_kwargs=cross_attention_kwargs
            )
            eps_R = self._predict_noise_single_frame(
                xR, t_eff, odd_target_index(k+1),
                full_embeddings,
                do_classifier_free_guidance, guidance_scale,
                cross_attention_kwargs=cross_attention_kwargs
            )

            x0_L = (xL - sqrt_omb * eps_L) / (sqrt_ab + 1e-8)
            x0_R = (xR - sqrt_omb * eps_R) / (sqrt_ab + 1e-8)

            # ---- 噪声插值：相关性归一化（高维近似 slerp）----
            eps_L_flat = eps_L.view(B, -1)
            eps_R_flat = eps_R.view(B, -1)
            d = eps_L_flat.shape[-1]
            rho = (eps_L_flat * eps_R_flat).sum(dim=-1, keepdim=True) / max(d, 1)
            rho = rho.clamp(-0.999, 0.999).view(B, 1, 1, 1)

            eps_interp_raw = lam * eps_L + (1 - lam) * eps_R
            norm2 = lam**2 + (1 - lam)**2 + 2 * lam * (1 - lam) * rho
            eps_interp = eps_interp_raw / torch.sqrt(torch.clamp(norm2, min=1e-8))
            # 
            # ---- 数据项插值（可替换为取一侧以更“稳”）----
            x0_interp = lam * x0_L + (1 - lam) * x0_R

            # 合成偶数帧 x_t
            x_even = sqrt_ab * x0_interp + sqrt_omb * eps_interp  # [B,C,H,W]

            # ----（可选）投影回 q_t：去噪一步，再正向加噪一步 ----
            if project_back_to_qt and t_eff > 0:
                eps_even = self._predict_noise_single_frame(
                    x_even, t_eff, odd_target_index(k) + 1,
                    full_embeddings,
                    do_classifier_free_guidance, guidance_scale,
                    cross_attention_kwargs=cross_attention_kwargs
                )
                # 反向一步： t -> t-1
                x_prev = ddpm_scheduler.step(
                    eps_even, torch.tensor(t_eff, device=device, dtype=torch.long), x_even, generator=generator
                ).prev_sample
                # 正向一步： t-1 -> t
                if isinstance(generator, list):
                    z = torch.randn_like(x_even)  # 简化：不逐帧选 generator
                else:
                    z = torch.randn(x_even.shape, generator=generator, device=device, dtype=x_even.dtype)
                x_even = sqrt_alpha_step * x_prev + sqrt_one_minus_alpha_step * z

            frames_out.append(x_even.unsqueeze(2))

        # 追加最后一个奇数帧
        frames_out.append(half_noise_latents[:, :, -1:])

        result = torch.cat(frames_out, dim=2)  # [B,C,2F_odd-1,H,W]
        print(f"[INFO] Strict noise interpolation done: {half_noise_latents.shape} -> {result.shape}")
        return result


    def _generate_full_embeddings_for_interpolation(
        self, start_embeds, end_embeds, add_text_embeds, add_time_ids,
        video_length, do_classifier_free_guidance, interpolation_strategy, device,
        semantic_points=None, power_fast_exponent: float = 0.4,
    ):
        """为插值生成完整视频长度的embedding（偶数帧将用到）"""
        if do_classifier_free_guidance:
            expected_cfg_size = start_embeds.shape[0] // 2
            start_neg, start_pos = start_embeds[:expected_cfg_size], start_embeds[expected_cfg_size:]
            end_neg,   end_pos   = end_embeds[:expected_cfg_size],   end_embeds[expected_cfg_size:]

            full_pos = self.interpolate_embeddings_for_video(
                start_embeds=start_pos, end_embeds=end_pos,
                video_length=video_length, interpolation_strategy=interpolation_strategy, device=device,
                semantic_points=semantic_points, power_fast_exponent=power_fast_exponent
            )
            full_neg = self.interpolate_embeddings_for_video(
                start_embeds=start_neg, end_embeds=end_neg,
                video_length=video_length, interpolation_strategy=interpolation_strategy, device=device,
                semantic_points=semantic_points, power_fast_exponent=power_fast_exponent
            )
            full_prompt_embeds = torch.cat([full_neg, full_pos], dim=0)

            cfg_chunk_size = add_text_embeds.shape[0] // 2
            neg_text, pos_text = add_text_embeds[:cfg_chunk_size], add_text_embeds[cfg_chunk_size:]
            neg_time, pos_time = add_time_ids[:cfg_chunk_size],   add_time_ids[cfg_chunk_size:]

            full_text_embeds = torch.cat([
                neg_text.repeat(video_length, 1),
                pos_text.repeat(video_length, 1)
            ], dim=0)
            full_time_ids = torch.cat([
                neg_time.repeat(video_length, 1),
                pos_time.repeat(video_length, 1)
            ], dim=0)
        else:
            full_prompt_embeds = self.interpolate_embeddings_for_video(
                start_embeds=start_embeds, end_embeds=end_embeds,
                video_length=video_length, interpolation_strategy=interpolation_strategy, device=device,
                semantic_points=semantic_points, power_fast_exponent=power_fast_exponent
            )
            full_text_embeds = add_text_embeds.repeat(video_length, 1)
            full_time_ids    = add_time_ids.repeat(video_length, 1)

        return {
            'prompt_embeds': full_prompt_embeds,
            'text_embeds':   full_text_embeds,
            'time_ids':      full_time_ids,
        }


    def _predict_noise_single_frame(
        self, x_t, timestep, frame_idx, full_embeddings,
        do_classifier_free_guidance, guidance_scale,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        """为单帧预测噪声（CFG 批维对齐修复）。x_t: [B,C,H,W]，返回 [B,C,H,W]"""
        B = x_t.shape[0]

        if do_classifier_free_guidance:
            V = full_embeddings['prompt_embeds'].shape[0] // 2  # 视频长度
            frame_embed_neg = full_embeddings['prompt_embeds'][frame_idx:frame_idx+1]
            frame_embed_pos = full_embeddings['prompt_embeds'][frame_idx+V:frame_idx+V+1]
            frame_text_neg  = full_embeddings['text_embeds'][frame_idx:frame_idx+1]
            frame_text_pos  = full_embeddings['text_embeds'][frame_idx+V:frame_idx+V+1]
            frame_time_neg  = full_embeddings['time_ids'][frame_idx:frame_idx+1]
            frame_time_pos  = full_embeddings['time_ids'][frame_idx+V:frame_idx+V+1]

            frame_embedding   = torch.cat([frame_embed_neg, frame_embed_pos], dim=0).repeat_interleave(B, dim=0)   # [2B, D]
            frame_text_embeds = torch.cat([frame_text_neg, frame_text_pos], dim=0).repeat_interleave(B, dim=0)     # [2B, D]
            frame_time_ids    = torch.cat([frame_time_neg, frame_time_pos], dim=0).repeat_interleave(B, dim=0)     # [2B, D]

            x_t_input = torch.cat([x_t, x_t], dim=0)  # [2B, C, H, W]
        else:
            frame_embedding   = full_embeddings['prompt_embeds'][frame_idx:frame_idx+1].repeat_interleave(B, dim=0)
            frame_text_embeds = full_embeddings['text_embeds'][frame_idx:frame_idx+1].repeat_interleave(B, dim=0)
            frame_time_ids    = full_embeddings['time_ids'][frame_idx:frame_idx+1].repeat_interleave(B, dim=0)
            x_t_input = x_t

        t_tensor = torch.tensor(timestep, device=x_t.device, dtype=torch.long)
        self.update_attention_context(timestep)  # 与主循环一致地刷新上下文

        with torch.no_grad():
            added_cond_kwargs = {"text_embeds": frame_text_embeds, "time_ids": frame_time_ids}
            eps_pred = self.unet(
                x_t_input, t_tensor,
                encoder_hidden_states=frame_embedding,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        if do_classifier_free_guidance:
            eps_uncond, eps_text = eps_pred.chunk(2)
            eps_pred = eps_uncond + guidance_scale * (eps_text - eps_uncond)

        return eps_pred  # [B, C, H, W]


    def _ddim_attention_refinement_hybrid_with_interpolation(
        self,
        half_noise_latents: torch.FloatTensor,
        t_start: int,
        ddim_steps: int,
        # 新增：插值参数
        start_embeds: torch.FloatTensor,
        end_embeds: torch.FloatTensor,
        
        add_time_ids: torch.FloatTensor,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        add_text_embeds: torch.FloatTensor,
        interpolation_strategy: str = "linear",
        guidance_rescale: float = 0.0,

        text_consistency_strength: float = 0.8,
        text_consistency_active_layers: Optional[List[str]] = None,
        causal_layers: Optional[List[str]] = None,
        causal_reference_frame_idx: int = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        semantic_points=None,
        power_fast_exponent: float = 0.4,
    ) -> torch.FloatTensor:
        """DDIM 注意力精细化阶段：混合注意力+embedding插值版本"""
        print(f"[INFO] Starting DDIM hybrid attention refinement with embedding interpolation")
        
        batch_size, c, video_length, h, w = half_noise_latents.shape
        device = half_noise_latents.device
        
        # 1. 正确处理CFG的embedding插值
        if do_classifier_free_guidance:
            # 检查start_embeds是否包含CFG结构
            expected_cfg_size = start_embeds.shape[0] // 2
            if expected_cfg_size == 0:
                print(f"[ERROR] CFG enabled but start_embeds only has {start_embeds.shape[0]} batch, expected 2+")
                raise ValueError("CFG is enabled but embeddings don't contain CFG structure")
            
            # start_embeds和end_embeds已经包含[negative, positive]结构
            start_neg, start_pos = start_embeds[:expected_cfg_size], start_embeds[expected_cfg_size:]
            end_neg, end_pos = end_embeds[:expected_cfg_size], end_embeds[expected_cfg_size:]
            
            print(f"[DEBUG] CFG splitting: start_embeds {start_embeds.shape} -> neg {start_neg.shape}, pos {start_pos.shape}")
            
            # 分别插值positive和negative
            interpolated_pos = self.interpolate_embeddings_for_video(
                start_embeds=start_pos,
                end_embeds=end_pos,
                video_length=video_length,
                interpolation_strategy=interpolation_strategy,
                device=device,
                semantic_points=semantic_points,
                power_fast_exponent=power_fast_exponent,
            )
            
            interpolated_neg = self.interpolate_embeddings_for_video(
                start_embeds=start_neg,
                end_embeds=end_neg,
                video_length=video_length,
                interpolation_strategy=interpolation_strategy,
                device=device,
                semantic_points=semantic_points,
                power_fast_exponent=power_fast_exponent,
            )
            
            # 重新组合为CFG格式：[neg_frames, pos_frames]
            interpolated_prompt_embeds = torch.cat([interpolated_neg, interpolated_pos], dim=0)
            print(f"[DEBUG] CFG recombined: interpolated_prompt_embeds {interpolated_prompt_embeds.shape}")
        else:
            # 无CFG时直接插值
            interpolated_prompt_embeds = self.interpolate_embeddings_for_video(
                start_embeds=start_embeds,
                end_embeds=end_embeds,
                video_length=video_length,
                interpolation_strategy=interpolation_strategy,
                device=device,
                semantic_points=semantic_points,
                power_fast_exponent=power_fast_exponent,
            )
        
        # 2. 设置DDIM调度器
        self.scheduler.set_timesteps(ddim_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 过滤时间步
        ddim_timesteps = [t for t in timesteps if int(t.item()) <= t_start]
        if not ddim_timesteps:
            ddim_timesteps = timesteps.tolist()
        
        print(f"[INFO] DDIM processing {len(ddim_timesteps)} steps, {video_length} frames")
        
        # 设置因果注意力的源映射
        current_latents = rearrange(half_noise_latents, "b c f h w -> (b f) c h w")
        
        # ===== 关键：设置注意力策略 =====
        # 在相对低噪声状态下使用注意力机制
        active_timesteps = [int(t.item()) for t in ddim_timesteps[:50]]
        
        self.setup_selective_consistency_attention(
            unet_chunk_size=2 if do_classifier_free_guidance else 1,
            text_layers=text_consistency_active_layers ,
            text_consistency_strength=text_consistency_strength,
            causal_layers=causal_layers ,
            causal_reference_frame_idx=causal_reference_frame_idx,
            active_timesteps=active_timesteps,
        )
        
        # 设置注意力源映射
     
        
        # 4. 重排为UNet格式
        current_latents = rearrange(half_noise_latents, "b c f h w -> (f b) c h w")
        
        # 5. 扩展其他条件
        if do_classifier_free_guidance:
    # add_text_embeds 和 add_time_ids 在 __call__ 中已经包含CFG结构 [neg, pos]
    # 只需要重复每个条件以匹配video_length
            cfg_chunk_size = add_text_embeds.shape[0] // 2
            
            # 分离负面和正面条件
            neg_text_embeds, pos_text_embeds = add_text_embeds[:cfg_chunk_size], add_text_embeds[cfg_chunk_size:]
            neg_time_ids, pos_time_ids = add_time_ids[:cfg_chunk_size], add_time_ids[cfg_chunk_size:]
            
            # 扩展到video_length帧
            expanded_neg_text_embeds = neg_text_embeds.repeat(video_length, 1)
            expanded_pos_text_embeds = pos_text_embeds.repeat(video_length, 1)
            expanded_neg_time_ids = neg_time_ids.repeat(video_length, 1)
            expanded_pos_time_ids = pos_time_ids.repeat(video_length, 1)
            
            # 重新组合为CFG格式
            expanded_add_text_embeds = torch.cat([expanded_neg_text_embeds, expanded_pos_text_embeds], dim=0)
            expanded_add_time_ids = torch.cat([expanded_neg_time_ids, expanded_pos_time_ids], dim=0)
            
            print(f"[DEBUG] CFG conditions: add_text_embeds {add_text_embeds.shape} -> expanded {expanded_add_text_embeds.shape}")
        else:
            # 无CFG时直接扩展
            expanded_add_text_embeds = add_text_embeds.repeat(video_length, 1)
            expanded_add_time_ids = add_time_ids.repeat(video_length, 1)

        # 6. DDPM去噪循环 - 修复CFG处理
        print(f"[INFO] Running DDiM with interpolated embeddings")

        with self.progress_bar(total=len(ddim_timesteps)) as progress_bar:
            for step_idx, t in enumerate(ddim_timesteps):
                self.update_attention_context(int(t.item()))
                # CFG扩展 - 修正这里，不要重复扩展已经包含CFG结构的条件
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([current_latents] * 2)
                    # 所有输入都已经是正确的CFG格式，直接使用
                    prompt_embeds_input = interpolated_prompt_embeds  # [16, 77, 2048]
                    text_embeds_input = expanded_add_text_embeds      # [16, projection_dim]
                    time_ids_input = expanded_add_time_ids            # [16, 6]
                else:
                    latent_model_input = current_latents
                    prompt_embeds_input = interpolated_prompt_embeds
                    text_embeds_input = expanded_add_text_embeds
                    time_ids_input = expanded_add_time_ids
                
                # 调试信息
                if step_idx == 0:  # 只在第一步打印调试信息
                    print(f"[DEBUG] Step {step_idx}: latent_input {latent_model_input.shape}, prompt_embeds {prompt_embeds_input.shape}, text_embeds {text_embeds_input.shape}, time_ids {time_ids_input.shape}")
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # UNet预测
                added_cond_kwargs = {
                    "text_embeds": text_embeds_input,
                    "time_ids": time_ids_input
                }
                
                t_tensor = torch.as_tensor(t, device=device, dtype=torch.long)

                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t_tensor,
                        encoder_hidden_states=prompt_embeds_input,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                
                # CFG
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # DDPM step
                current_latents = self.scheduler.step(
                    noise_pred, t_tensor, current_latents, generator=generator
                ).prev_sample
                
                # Callbacks
                progress_bar.update()
                if callback is not None and step_idx % callback_steps == 0:
                    callback_latents = rearrange(current_latents, "(f b) c h w -> b c f h w", f=video_length, b=batch_size)
                    callback(step_idx, t, callback_latents)
        
        # 重排回视频格式
        result = rearrange(current_latents, "(f b) c h w -> b c f h w", f=video_length, b=batch_size)
        print(f"[INFO] DDIM hybrid attention + embedding interpolation completed: {result.shape}")
        return result


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        video_length: int = 8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 12.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        # 其他标准参数...
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        # 混合注意力参数
        progressive_prompts: Optional[Dict[str, str]] = {"start":"a girl in blue is standing by a sea","end":"a girl in blue is dancing by a sea"},
        progressive_interpolation: str = "cosine",
        # 新策略：LLM 语义时间点 + power_fast 偏置矫正
        semantic_points: Optional[List[float]] = None,
        power_fast_exponent: float = 0.4,
        ddpm_causal_layers: Optional[List[str]] = None,
        ddim_causal_layers: Optional[List[str]] = None,
        # 混合采样参数
        init_mode: str = "ddpm_structure_ddim_refine",
        ddpm_ratio: float = 0.03,
        ddmp_steps: int = 30,
        ddim_steps: int = 100,
        text_consistency_strength: float = 0.8,
        text_consistency_layers: Optional[List[str]] = None,
        causal_reference_frame_idx: int = 0,
        **kwargs
        ):
        """混合注意力+embedding插值支持版本"""
        
        # 0. 默认参数设置
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. 检查渐变提示词参数
        progressive_enabled = progressive_prompts is not None
        if progressive_enabled:
            if not isinstance(progressive_prompts, dict) or "start" not in progressive_prompts or "end" not in progressive_prompts:
                raise ValueError("progressive_prompts must be a dict with 'start' and 'end' keys")
            
            start_prompt = progressive_prompts["start"]
            end_prompt = progressive_prompts["end"]
            print(f"[INFO] Progressive prompts enabled:")
            print(f"  Start: '{start_prompt}'")
            print(f"  End: '{end_prompt}'")
        else:
            print("[INFO] Standard mode (no progressive prompts)")
            # 使用标准prompt作为start prompt（保持兼容性）
            start_prompt = prompt if isinstance(prompt, str) else prompt[0] if prompt else ""
            end_prompt = start_prompt

        # 2. 输入检查
        check_prompt = start_prompt
        self.check_inputs(
            check_prompt, prompt_2, height, width, callback_steps,
            negative_prompt, negative_prompt_2, None, None, None, None,
        )

        # 3. 批次大小
        batch_size = 1  # 简化为单批次
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3.1 校验 LLM 语义时间点（若提供）
        if semantic_points is not None:
            semantic_points = [float(x) for x in semantic_points]
            if len(semantic_points) < 2:
                raise ValueError(f"semantic_points must contain at least 2 points, got {len(semantic_points)}")
            if len(semantic_points) != video_length:
                raise ValueError(
                    f"semantic_points length ({len(semantic_points)}) must equal "
                    f"video_length ({video_length}); one point per frame, no resampling."
                )
            print(f"[INFO] Semantic-point interpolation ENABLED: s={semantic_points}")
            print(f"[INFO]   -> will be mapped by power_fast (exp={power_fast_exponent}), "
                  f"progressive_interpolation='{progressive_interpolation}' is IGNORED")
        

        # 4. 编码提示词
        if progressive_enabled:
            # 渐变模式：编码 start 和 end 嵌入
            start_embeds_raw, end_embeds_raw = self.encode_simple_progressive_prompts(
                start_prompt=start_prompt,
                end_prompt=end_prompt,
                device=device,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )
            
            print(f"[DEBUG] Progressive mode: start_embeds_raw {start_embeds_raw.shape}, end_embeds_raw {end_embeds_raw.shape}")
            
            # progressive模式下，start_embeds_raw和end_embeds_raw已经包含CFG结构，直接使用
            start_embeds = start_embeds_raw
            end_embeds = end_embeds_raw
            
            # 为了保持兼容性，设置prompt_embeds（但不会在后续CFG处理中使用）
            if do_classifier_free_guidance:
                prompt_embeds = start_embeds_raw  # 已经包含CFG结构
                negative_prompt_embeds = None    # 设为None避免重复处理
            else:
                prompt_embeds = start_embeds_raw
                negative_prompt_embeds = None
            
            # 获取 pooled embeddings（使用 start prompt）
            _, _, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompt=start_prompt, prompt_2=prompt_2, device=device,
                num_images_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
            )
        else:
            # 标准模式
            (
                prompt_embeds, negative_prompt_embeds,
                pooled_prompt_embeds, negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt, prompt_2=prompt_2, device=device,
                num_images_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
            )
            start_embeds = prompt_embeds
            end_embeds = prompt_embeds

        # 5. 准备额外条件
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype)

# CFG处理 - 修复这里
        if do_classifier_free_guidance:
            if progressive_enabled:
                # Progressive模式：embeddings已经包含CFG结构，只处理其他条件
                # prompt_embeds 和 start_embeds/end_embeds 已经是正确的CFG格式
                pass  # 不需要额外处理prompt_embeds
            else:
                # 标准模式：需要合并negative和positive embeddings
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                start_embeds = prompt_embeds
                end_embeds = prompt_embeds
            
            # 处理其他条件（两种模式都需要）
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_videos_per_prompt, 1)
           
        

        # 7. 准备初始潜在变量
        num_channels_latents = self.unet.config.in_channels
        
        if latents is None:
            latent_height = height // self.vae_scale_factor
            latent_width = width // self.vae_scale_factor
            
            single_frame_latents = randn_tensor(
                (batch_size * num_videos_per_prompt, num_channels_latents, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=prompt_embeds.dtype,
            )
            single_frame_latents = single_frame_latents * self.scheduler.init_noise_sigma
        else:
            single_frame_latents = latents[:, :, 0, :, :] if len(latents.shape) == 5 else latents

        # 8. 混合采样流程
        if init_mode == "ddpm_structure_ddim_refine":
            print(f"[INFO] Using embedding interpolation + hybrid attention mixed sampling")
            
            # 计算切换时间步
            t_switch = self._calculate_switch_timestep(ddpm_ratio, 1000)
            
            # 阶段1: DDPM结构建立 - 带embedding插值
            print(f"[INFO] Phase 1: DDPM structure building with embedding interpolation")
            half_noise_latents = self._ddpm_structure_building_with_interpolation(
                single_frame_latents=single_frame_latents,
                video_length=video_length,
                ddpm_steps=ddmp_steps,
                t_start=t_switch,
                # 插值参数
                start_embeds=start_embeds,
                add_text_embeds=add_text_embeds,
                end_embeds=end_embeds,
                interpolation_strategy="cosine",  # DDPM用简单策略（semantic_points 提供时将被忽略）
                causal_layers=ddpm_causal_layers,
                add_time_ids=add_time_ids,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=guidance_scale,
                generator=generator,
                cross_attention_kwargs=cross_attention_kwargs,
                callback=callback,
                callback_steps=callback_steps,
                semantic_points=semantic_points,
                power_fast_exponent=power_fast_exponent,
            )
            
            # 阶段2: DDIM混合注意力精细化 - 带embedding插值
            print(f"[INFO] Phase 2: DDIM hybrid attention refinement with embedding interpolation")
            final_latents = self._ddim_attention_refinement_hybrid_with_interpolation(
                half_noise_latents=half_noise_latents,
                t_start=t_switch,
                ddim_steps=ddim_steps,
                # 插值参数
                start_embeds=start_embeds,
                end_embeds=end_embeds,
                interpolation_strategy=progressive_interpolation,  
                # 其他参数
                add_text_embeds=add_text_embeds,
                add_time_ids=add_time_ids,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                generator=generator,
                text_consistency_strength=text_consistency_strength,
                text_consistency_active_layers=text_consistency_layers,
                causal_layers=ddim_causal_layers,
                causal_reference_frame_idx=causal_reference_frame_idx,
                cross_attention_kwargs=cross_attention_kwargs,
                callback=callback,
                callback_steps=callback_steps,
                semantic_points=semantic_points,
                power_fast_exponent=power_fast_exponent,
            )
            
            latents = final_latents
        else:
            # 直接 DDIM：生成 (B, C, T, H, W) 的独立初始噪声，然后一次性丢进 DDIM 精炼
            print("[INFO] Direct DDIM sampling (no DDPM warmup)")

            latent_height = height // self.vae_scale_factor
            latent_width = width // self.vae_scale_factor

            # 如果外部没传 latents，就自己按帧生成；传了就按形状兜底处理成 5D
            if latents is None:
                half_noise_latents = randn_tensor(
                    (
                        batch_size * num_videos_per_prompt,
                        num_channels_latents,
                        video_length,
                        latent_height,
                        latent_width,
                    ),
                    generator=generator,
                    device=device,
                    dtype=prompt_embeds.dtype,
                ) * self.scheduler.init_noise_sigma
            else:
                # 支持外部传入的 4D/5D 潜变量
                if latents.dim() == 5:
                    half_noise_latents = latents.to(device=device, dtype=prompt_embeds.dtype)
                elif latents.dim() == 4:
                    half_noise_latents = latents[:, :, None, :, :].to(device=device, dtype=prompt_embeds.dtype)
                else:
                    raise ValueError("latents must be (B,C,H,W) or (B,C,T,H,W)")

            # 直接用你的 DDIM 精炼（t_start=None 表示从完整 DDIM 计划开始）
            final_latents = self._ddim_attention_refinement_hybrid_with_interpolation(
                half_noise_latents=half_noise_latents,
                t_start=999,
                ddim_steps=ddim_steps,
                # 插值参数
                start_embeds=start_embeds,
                end_embeds=end_embeds,
                interpolation_strategy=progressive_interpolation,
                # 其他参数
                add_text_embeds=add_text_embeds,
                add_time_ids=add_time_ids,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                generator=generator,
                text_consistency_strength=text_consistency_strength,
                text_consistency_active_layers=text_consistency_layers,
                causal_layers=ddim_causal_layers,
                causal_reference_frame_idx=causal_reference_frame_idx,
                cross_attention_kwargs=cross_attention_kwargs,
                callback=callback,
                callback_steps=callback_steps,
                semantic_points=semantic_points,
                power_fast_exponent=power_fast_exponent,
            )

            latents = final_latents
            
            

        # 9. 解码和后处理
        if not output_type == "latent":
            image = self.decode_video_latents(latents)
            has_nsfw_concept = None
            image = rearrange(image, "b c f h w -> (b f) h w c")
        else:
            image = latents
            has_nsfw_concept = None

        # 10. 清理
        self.restore_original_attention()
            
        if hasattr(self, "maybe_free_model_hooks"):
            self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return FuVideoPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)