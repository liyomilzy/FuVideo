import torch
from enum import Enum
import gc
import logging

from diffusers.schedulers import DDIMScheduler
from .text_to_video_pipeline import FuVideoPipeline

import utils
import os

on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"


class ModelType(Enum):
    FuVideo = 1


class Model:
    def __init__(self, device, dtype, **kwargs):
        # Device validation
        if device is None or device == "None":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.generator = torch.Generator(device=self.device)

        # Simple logging setup
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.pipe = None
        self.model_type = None
        self.model_name = ""

    def set_model(self, model_id: str, **kwargs):
        """Set and load model"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

        safety_checker = kwargs.pop('safety_checker', None)
        
        self.logger.info(f"[Model] Loading model: {model_id}")
        
        # Load model
        self.pipe = FuVideoPipeline.from_pretrained(
            model_id, 
            safety_checker=safety_checker, 
            torch_dtype=self.dtype,
            **kwargs
        )
        
        # Move to target device
        self.pipe = self.pipe.to(self.device)
        self.logger.info(f"[Model] Pipeline loaded on device: {self.device}")
        
        self.model_type = ModelType.FuVideo
        self.model_name = model_id

    def process_text2video(self,
                           prompt,
                           model_name="stabilityai/stable-diffusion-xl-base-1.0",
                           negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic",
                           video_length=8,
                           seed=42,
                           resolution=512,
                           fps=8,
                           guidance_scale=13,
                           guidance_rescale=0.0,
                           
                           # Progressive prompt parameters
                           progressive_prompts=None,
                           progressive_interpolation="cosine",
                           # LLM 语义时间点 + power_fast 偏置矫正
                           semantic_points=None,
                           power_fast_exponent=0.4,
                           
                           # Mixed sampling parameters  
                           init_mode="ddpm_structure_ddim_refine",
                           ddpm_ratio=0.03,
                           ddmp_steps=30,  # Note: keeping original typo from pipeline
                           ddim_steps=100,
                           
                           # Attention layer parameters - THE TWO KEY PARAMETERS
                           text_consistency_strength=0.8,
                           text_consistency_layers=None,
                           ddpm_causal_layers=None,  # DDPM stage causal attention layers
                           ddim_causal_layers=None,  # DDIM stage causal attention layers  
                           causal_reference_frame_idx=0,
                           
                           # SDXL parameters
                           original_size=None,
                           target_size=None,
                           
                           # Output parameters
                           watermark='NKU',
                           path=None,
                           exp_name=None,
                           save_frames=False,
                           **kwargs):
        """Process text-to-video generation with all pipeline parameters including ddpm_causal_layers and ddim_causal_layers"""
        
        self.logger.info(f"[FuVideo] Starting generation")
        self.logger.info(f"[FuVideo] Mode: {init_mode}")
        self.logger.info(f"[FuVideo] DDPM: {ddpm_ratio*100:.1f}%/{ddmp_steps} steps, DDIM: {ddim_steps} steps")
        
        # Handle progressive prompts
        if progressive_prompts:
            self.logger.info(f"[FuVideo] Progressive mode: '{progressive_prompts['start']}' -> '{progressive_prompts['end']}'")
        else:
            self.logger.info(f"[FuVideo] Standard mode: '{prompt}'")

        # Load model if needed
        if self.model_type != ModelType.FuVideo or model_name != self.model_name:
            self.logger.info("[FuVideo] Loading model...")
            self.set_model(model_id=model_name, use_safetensors=True)
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Setup generator
        if self.generator.device != self.device:
            self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        # Enhance prompts
        added_prompt = "high quality, HD, 8K, trending on artstation, high focus, dramatic lighting"
        
        if isinstance(prompt, str) and prompt.strip():
            prompt = prompt.strip().rstrip(',.')
            enhanced_prompt = f"{prompt}, {added_prompt}"
        else:
            enhanced_prompt = added_prompt

        if progressive_prompts:
            start_prompt = progressive_prompts["start"].strip().rstrip(',.')
            end_prompt = progressive_prompts["end"].strip().rstrip(',.')
            enhanced_progressive = {
                "start": f"{start_prompt}, {added_prompt}",
                "end": f"{end_prompt}, {added_prompt}"
            }
        else:
            enhanced_progressive = None

        # Set default negative prompt if empty
        if not negative_prompt.strip():
            negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"

        # Set size parameters
        if original_size is None:
            original_size = (resolution, resolution)
        if target_size is None:
            target_size = (resolution, resolution)

        # Set default attention layers if not provided
        if text_consistency_layers is None:
            text_consistency_layers = []
        
        if ddpm_causal_layers is None:
            ddpm_causal_layers = []
            
        if ddim_causal_layers is None:
            ddim_causal_layers = []

        self.logger.info(f"[FuVideo] Attention layers - Text consistency: {text_consistency_layers}")
        self.logger.info(f"[FuVideo] Attention layers - DDPM causal: {ddpm_causal_layers}")
        self.logger.info(f"[FuVideo] Attention layers - DDIM causal: {ddim_causal_layers}")

        try:
            # Generate video with all pipeline parameters - INCLUDING THE TWO KEY PARAMETERS
            result = self.pipe(
                prompt=enhanced_prompt if not enhanced_progressive else None,
                negative_prompt=negative_prompt,
                video_length=video_length,
                height=resolution,
                width=resolution,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                generator=self.generator,
                output_type='np',
                return_dict=True,
                
                # Progressive prompts parameters
                progressive_prompts=enhanced_progressive,
                progressive_interpolation=progressive_interpolation,
                # LLM 语义时间点 + power_fast 偏置矫正
                semantic_points=semantic_points,
                power_fast_exponent=power_fast_exponent,
                
                # Mixed sampling parameters
                init_mode=init_mode,
                ddpm_ratio=ddpm_ratio,
                ddmp_steps=ddmp_steps,  # Note: keeping original typo from pipeline
                ddim_steps=ddim_steps,
                
                # Attention layer parameters - THE CRITICAL ONES YOU ASKED FOR
                text_consistency_strength=text_consistency_strength,
                text_consistency_layers=text_consistency_layers,
                ddpm_causal_layers=ddpm_causal_layers,  # ← THIS ONE
                ddim_causal_layers=ddim_causal_layers,  # ← THIS ONE  
                causal_reference_frame_idx=causal_reference_frame_idx,
                
                # SDXL parameters
                original_size=original_size,
                target_size=target_size,
                
                **kwargs
            )

            frames = result.images
            self.logger.info(f"[FuVideo] Generated {frames.shape} frames, range: [{frames.min():.3f}, {frames.max():.3f}]")

            # Check for dark frames
            if frames.max() < 0.01:
                self.logger.warning("[FuVideo] Generated frames appear very dark!")

            # Create video
            watermark_path = None
            if watermark and watermark != 'NKU' and os.path.exists(watermark):
                watermark_path = watermark
            
            video_path = utils.create_video(frames, fps, path=path, watermark=watermark_path, exp_name=exp_name, save_frames=save_frames)
            self.logger.info(f"[FuVideo] Video saved to: {video_path}")
            
            return frames, video_path

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def cleanup(self):
        """Clean up resources"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("[Model] Cleanup completed")

    def enable_memory_optimization(self):
        """Enable memory optimization"""
        if self.pipe is not None:
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            if hasattr(self.pipe, 'enable_model_cpu_offload'):
                self.pipe.enable_model_cpu_offload()
            self.logger.info("[Model] Memory optimization enabled")