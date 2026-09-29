"""
FuVideo: zero-shot text-to-video generation script
Supports progressive prompts and mixed DDPM/DDIM sampling with full attention layer control
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
from model import Model


def generate_experiment_name(args):
    """
    修复版本：生成简洁且安全的实验名称
    防止文件名过长导致FFmpeg错误
    包含init_mode参数
    """
    import hashlib
    
    # 层名称简化映射
    LAYER_MAPPING = {
        'down_blocks.0': 'd0',
        'down_blocks.1': 'd1', 
        'down_blocks.2': 'd2',
        'mid_block': 'mid',
        'up_blocks.0': 'u0',
        'up_blocks.1': 'u1',
        'up_blocks.2': 'u2'
    }
    
    # init_mode简化映射
    INIT_MODE_MAPPING = {
        'ddpm_structure_ddim_refine': 'ddpm_str',
        'just_ddim': 'j_ddim'
    }
    
    def safe_simplify_layers(layer_list):
        """安全地简化层名称列表"""
        if not layer_list:
            return "none"
        
        # 处理连接字符串的情况（你脚本中的写法）
        if isinstance(layer_list, str):
            # 手动解析连接的层名称字符串
            layers = []
            temp_str = layer_list
            
            # 依次查找并提取已知的层名称
            known_layers = ['down_blocks.0', 'down_blocks.1', 'down_blocks.2', 
                           'mid_block', 'up_blocks.0', 'up_blocks.1', 'up_blocks.2']
            
            for known_layer in known_layers:
                if known_layer in temp_str:
                    layers.append(known_layer)
                    temp_str = temp_str.replace(known_layer, '', 1)  # 移除已找到的
            
            layer_list = layers
        
        # 如果layer_list不是列表，尝试转换
        if not isinstance(layer_list, (list, tuple)):
            return "none"
        
        simplified = []
        for layer in layer_list:
            if isinstance(layer, str) and layer.strip():
                simplified_name = LAYER_MAPPING.get(layer.strip(), layer.strip()[:3])
                if simplified_name not in simplified:
                    simplified.append(simplified_name)
        
        if not simplified:
            return "none"
        
        # 排序并连接，严格限制长度
        result = "_".join(sorted(simplified)[:5])  # 最多5个层名
        return result[:30]  # 限制总长度为30字符
    
    # 构建简洁的实验名称
    parts = []
    
    # 基础标识
    if getattr(args, 'use_progressive', False):
        interp = getattr(args, 'progressive_interpolation', 'linear')
        parts.append(f"prog_{interp[:3]}")  # 截断插值方法名
    else:
        parts.append("std")
    
    # 添加init_mode参数 - 新增部分
    init_mode = getattr(args, 'init_mode', 'just_ddim')
    init_mode_short = INIT_MODE_MAPPING.get(init_mode, init_mode[:8])  # 默认截断到8字符
    parts.append(init_mode_short)
    
    # 采样参数（简化数字格式）
    ddpm_ratio = getattr(args, 'ddpm_ratio', 0.0)
    ddim_steps = getattr(args, 'ddim_steps', 100)
    
    parts.append(f"ddpm{int(ddpm_ratio*100):02d}")  # 03 -> 03
    parts.append(f"ddim{ddim_steps}")
    
    # 文本一致性
    txt_strength = getattr(args, 'text_consistency_strength', 0.8)
    parts.append(f"txt{int(txt_strength*10)}")  # 0.8 -> 8
    
    # 层配置（关键改进）
    ddpm_layers = getattr(args, 'ddpm_causal_layers', [])
    ddim_layers = getattr(args, 'ddim_causal_layers', [])
    
    ddpm_simplified = safe_simplify_layers(ddpm_layers)
    ddim_simplified = safe_simplify_layers(ddim_layers)
    
    if ddpm_simplified != "none":
        parts.append(f"dp_{ddpm_simplified}")
    
    if ddim_simplified != "none":
        parts.append(f"dm_{ddim_simplified}")
    
    # 连接所有部分
    exp_name = "_".join(parts)
    
    # 安全长度检查：如果太长，使用哈希
    if len(exp_name) > 100:  # 文件名长度限制
        # 保留前缀，用哈希替换详细配置
        prefix = "_".join(parts[:4])  # 保留基础信息包括init_mode
        config_hash = hashlib.md5(exp_name.encode()).hexdigest()[:8]
        exp_name = f"{prefix}_{config_hash}"
    
    return exp_name

# 在main函数中使用



def main():
    parser = argparse.ArgumentParser(description="Text-to-video generation with progressive prompts and attention control")
    
    # Basic parameters
    parser.add_argument("--prompt", type=str, 
                       default="A serene view of the sea under clear skies, with a few fluffy white clouds in the distance.",
                       help="Text prompt for video generation")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Model ID to use")
    parser.add_argument("--video_length", type=int, default=7,
                       help="Number of frames to generate")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Video resolution (width and height)")
    parser.add_argument("--fps", type=int, default=4,
                       help="Output video frame rate")
    parser.add_argument("--seed", type=int, default=54,
                       help="Random seed")
    parser.add_argument("--guidance_scale", type=float, default=12.5,
                       help="Guidance scale")
    parser.add_argument("--out", type=str, default=None,
                       help="Output video path")
    parser.add_argument("--save_frames", action="store_true",
                       help="Save individual frames as jpg images alongside the video")
    parser.add_argument("--n_prompt", type=str, default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic",
                       help="Negative prompt")

    # Progressive prompts
    parser.add_argument("--use_progressive", action=argparse.BooleanOptionalAction, default=True,
                       help="Enable progressive prompts mode (default: on; use --no-use_progressive to disable)")
    parser.add_argument("--start_prompt", type=str, 
                       default="A boy with an abdominal evisceration is standing by the seaside.",
                       help="Starting prompt")
    parser.add_argument("--end_prompt", type=str, 
                       default="A boy with an abdominal evisceration is dancing by the seaside.", 
                       help="Ending prompt")
    parser.add_argument("--progressive_interpolation", type=str, 
                       default="cosine",
                       choices=["linear", "cosine", "ease_in_out", "exponential",
                                "power_fast", "offset_cosine"],
                       help="Interpolation strategy (ignored if --semantic_points is provided)")

    # LLM 语义时间点 + power_fast 偏置矫正
    parser.add_argument("--semantic_points", type=float, nargs='+', default=None,
                       help="LLM-provided normalized semantic time points s=[s_0,...,s_{n-1}] "
                            "(s_0=0, s_{n-1}=1). Length must equal --video_length (one per frame, no resampling). "
                            "When provided, overrides --progressive_interpolation; "
                            "weights w_i = s_i ^ power_fast_exponent are used for embedding interpolation.")
    parser.add_argument("--power_fast_exponent", type=float, default=0.3,
                       help="Exponent for the power_fast bias correction applied to semantic points (default 0.4)")

    # Sampling parameters
    parser.add_argument("--init_mode", type=str, 
                       default="ddpm_structure_ddim_refine", choices=["ddpm_structure_ddim_refine","just_ddim"],
                       help="Denoise mode")

    parser.add_argument("--ddpm_ratio", type=float, default=0.03,
                       help="DDPM processing ratio (0.03 = first 3 percent)")
    parser.add_argument("--ddmp_steps", type=int, default=30,
                       help="DDPM steps")
    parser.add_argument("--ddim_steps", type=int, default=100,
                       help="DDIM steps")

    # Attention parameters - THE TWO KEY PARAMETERS YOU ASKED FOR
    parser.add_argument("--text_consistency_strength", type=float, default=0.8,
                       help="Text consistency strength")
    parser.add_argument("--text_consistency_layers", type=str, nargs='+',
                       default=[],
                       help="Layers for text consistency (e.g. 'down_blocks.1' 'mid_block')")
    parser.add_argument("--ddpm_causal_layers", type=str, nargs='+',
                       default=[],
                       help="Causal attention layers for DDPM stage")
    parser.add_argument("--ddim_causal_layers", type=str, nargs='+',
                       default=[],
                       help="Causal attention layers for DDIM stage")
    parser.add_argument("--causal_reference_frame_idx", type=int, default=0,
                       help="Reference frame index for causal attention")

    # Performance
    parser.add_argument("--enable_memory_opt", action="store_true",
                       help="Enable memory optimization")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=["float16", "float32"],
                       help="Data type")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    print("=" * 50)
    print("FuVideo Generation")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Video: {args.video_length} frames, {args.fps} FPS, {args.resolution}x{args.resolution}")
    print(f"Sampling: DDPM {args.ddpm_ratio*100:.1f}%/{args.ddmp_steps} + DDIM {args.ddim_steps}")
    
    # Handle progressive prompts
    if args.use_progressive:
        progressive_prompts = {"start": args.start_prompt, "end": args.end_prompt}
        print(f"Progressive: '{args.start_prompt}' -> '{args.end_prompt}'")
        if args.semantic_points is not None:
            print(f"Semantic points (LLM): {args.semantic_points}")
            print(f"power_fast exponent: {args.power_fast_exponent}  (progressive_interpolation ignored)")
        else:
            print(f"Interpolation: {args.progressive_interpolation}")
    else:
        progressive_prompts = None
        print(f"Prompt: '{args.prompt}'")
    
    # Show attention layer settings
    if args.text_consistency_layers:
        print(f"Text consistency layers: {args.text_consistency_layers}")
    if args.ddpm_causal_layers:
        print(f"DDPM causal layers: {args.ddpm_causal_layers}")
    if args.ddim_causal_layers:
        print(f"DDIM causal layers: {args.ddim_causal_layers}")
    
    print("=" * 50)
    exp_name = generate_experiment_name(args)
    # Initialize model
    model = Model(device=device, dtype=dtype)
    
    if args.enable_memory_opt:
        print("Enabling memory optimization...")
        model.enable_memory_optimization()
    
    try:
        # Generate video - WITH ALL PARAMETERS INCLUDING THE TWO YOU ASKED FOR
        frames, video_path = model.process_text2video(
            prompt=args.prompt,
            model_name=args.model,
            negative_prompt=args.n_prompt,
            video_length=args.video_length,
            resolution=args.resolution,
            fps=args.fps,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            path=args.out,
            exp_name=exp_name,
            # Progressive prompts
            progressive_prompts=progressive_prompts,
            progressive_interpolation=args.progressive_interpolation,
            # LLM 语义时间点 + power_fast 偏置矫正
            semantic_points=args.semantic_points,
            power_fast_exponent=args.power_fast_exponent,
            
            # Sampling
            init_mode=args.init_mode,
            ddpm_ratio=args.ddpm_ratio,
            ddmp_steps=args.ddmp_steps,  # Note: keeping original typo
            ddim_steps=args.ddim_steps,
            
            # Attention layers - THE CRITICAL PARAMETERS YOU REQUESTED
            text_consistency_strength=args.text_consistency_strength,
            text_consistency_layers=args.text_consistency_layers,
            ddpm_causal_layers=args.ddpm_causal_layers,  # ← THIS ONE
            ddim_causal_layers=args.ddim_causal_layers,  # ← THIS ONE
            causal_reference_frame_idx=args.causal_reference_frame_idx,
            save_frames=args.save_frames,
        )
        
        print("\n" + "=" * 50)
        print("Generation completed!")
        print(f"Frames: {frames.shape}")
        print(f"Video: {video_path}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Generation failed: {e}")
        raise
    
    finally:
        model.cleanup()

def show_examples():
    """Show usage examples with attention layer control"""
    examples = [
        {
            "name": "Basic Video",
            "cmd": "python main.py --prompt 'a beautiful mountain landscape' --video_length 16"
        },
        {
            "name": "Progressive Animation", 
            "cmd": "python main.py --use_progressive --start_prompt 'a cat sitting quietly' --end_prompt 'a cat playing with toys' --video_length 20 --progressive_interpolation cosine"
        },
        {
            "name": "High Quality with Custom Attention",
            "cmd": "python main.py --prompt 'majestic eagle soaring' --ddpm_ratio 0.05 --ddmp_steps 40 --ddim_steps 120 --resolution 1024 --text_consistency_strength 0.9"
        },
        {
            "name": "Full Attention Control (THE KEY PARAMETERS)",
            "cmd": "python main.py --use_progressive --start_prompt 'peaceful garden' --end_prompt 'blooming flowers everywhere' --ddpm_causal_layers down_blocks.1 mid_block --ddim_causal_layers down_blocks.2 up_blocks.1 --text_consistency_layers mid_block up_blocks.0"
        },
        {
            "name": "Different DDPM and DDIM Attention Layers",
            "cmd": "python main.py --prompt 'dancing girl by the sea' --ddpm_causal_layers down_blocks.0 down_blocks.1 --ddim_causal_layers mid_block up_blocks.0 up_blocks.1 --text_consistency_layers down_blocks.2 mid_block"
        }
    ]
    
    print("\nUsage Examples:")
    print("=" * 50)
    for example in examples:
        print(f"{example['name']}:")
        print(f"  {example['cmd']}")
        print()
    
    print("KEY ATTENTION LAYER PARAMETERS:")
    print("- text_consistency_layers: Controls text consistency in cross-attention")
    print("- ddpm_causal_layers: Controls causal attention during DDPM stage")  
    print("- ddim_causal_layers: Controls causal attention during DDIM stage")
    print()
    print("Available layers: down_blocks.0, down_blocks.1, down_blocks.2, mid_block, up_blocks.0, up_blocks.1, up_blocks.2")
    print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        show_examples()
    else:
        main()