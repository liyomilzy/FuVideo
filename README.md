# FuVideo

Zero-shot text-to-video generation built on [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and inspired by [Text2Video-Zero](https://arxiv.org/abs/2303.13439). Given a start prompt and an end prompt, it synthesizes a temporally coherent video by interpolating text embeddings frame-by-frame, with mixed ODE/SDE sampling and per-layer attention control.

## Key features

- **Mixed ODE/SDE sampling** (`init_mode=ddpm_structure_ddim_refine`): a short stochastic DDPM warm-up builds global structure, then a deterministic DDIM pass refines it. Only **odd frames** are denoised during the DDPM stage; even frames are synthesized by **noise-space interpolation** (correlation-normalized ε interpolation + linear x̂₀ interpolation, with an optional projection back to q<sub>t</sub>), cutting compute roughly in half.
- **Progressive prompts**: the start/end prompts are encoded separately and their embeddings are interpolated per frame with configurable curves — `linear`, `cosine`, `ease_in_out`, `power_fast`, `offset_cosine`.
- **LLM semantic timing** (`semantic_points`): instead of a fixed curve, an LLM outputs one normalized timing value per frame describing the physical rhythm of the change (`wᵢ = sᵢ^power_fast_exponent`). The `power_fast` exponent compensates the structural anchoring of causal self-attention on frame 0. See [llm_prompt_designer.md](llm_prompt_designer.md) and [system_prompt_for_paper_en.txt](system_prompt_for_paper_en.txt) for the LLM prompt design.
- **Layer-wise attention control**: cross-attention text-consistency (frames pulled toward a shared text-attention pattern) and causal self-attention alignment (K/V broadcast from the reference frame) can be enabled per UNet layer and per sampling stage (`text_consistency_layers`, `ddpm_causal_layers`, `ddim_causal_layers`).

## Pipeline overview

```
start prompt ──┐                                  ┌── frame 0 (start)
               ├─ encode → per-frame embedding    ├── frame 1  ⋮
end prompt   ──┘   lerp w/ curve or semantic pts  └── frame N (end)

Stage 1 (SDE): DDPM denoises odd frames only, t=999 → t_start (=1000·(1−ddpm_ratio))
Stage 1.5:     even frames = noise-space interpolation of neighboring odd frames @ t_start
Stage 2 (ODE): DDIM refines all frames, t_start → 0, with attention control on early steps
               → VAE decode → video
```

## Quick start

```bash
conda create -n t2v python=3.10 -y && conda activate t2v
pip install torch torchvision  # match your CUDA version
pip install -r requirements.txt
```

Then run (first run downloads `stabilityai/stable-diffusion-xl-base-1.0` from Hugging Face, ~7 GB):

```bash
python main.py \
  --start_prompt "A deflated red balloon lying flat on a white surface." \
  --end_prompt   "The balloon fully inflated, round and shiny, floating slightly above the surface." \
  --video_length 9 --resolution 1024 --fps 4 --seed 256
```

A single-command progressive generation with the recommended attention configuration:

```bash
python main.py \
  --start_prompt "a girl in blue is standing by a sea" \
  --end_prompt   "a girl in blue is dancing by a sea" \
  --video_length 9 --resolution 1024 --seed 256 \
  --ddpm_ratio 0.03 --ddmp_steps 30 --ddim_steps 100 \
  --text_consistency_strength 0.8 \
  --text_consistency_layers down_blocks.1 down_blocks.2 mid_block up_blocks.0 up_blocks.1 \
  --ddpm_causal_layers down_blocks.1 down_blocks.2 mid_block up_blocks.1 \
  --ddim_causal_layers down_blocks.1 down_blocks.2 up_blocks.1
```

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `--prompt` | — | Static prompt (used when progressive mode is off) |
| `--use_progressive` / `--no-use_progressive` | on | Progressive start→end mode |
| `--start_prompt`, `--end_prompt` | — | Progressive prompts |
| `--video_length` | 7 | Number of frames |
| `--resolution` | 512 | Frame size (square) |
| `--fps`, `--seed`, `--guidance_scale` | 4 / 54 / 12.5 | Video fps, seed, CFG scale |
| `--n_prompt` | longbody, lowres, … | Negative prompt |
| `--progressive_interpolation` | cosine | `linear` / `cosine` / `ease_in_out` / `power_fast` / `offset_cosine` (ignored when `--semantic_points` is given) |
| `--semantic_points` | None | LLM timing values, **one per frame** (length must equal `--video_length`); weights become `wᵢ = sᵢ^--power_fast_exponent` |
| `--power_fast_exponent` | 0.3 | Exponent of the power_fast bias correction |
| `--init_mode` | ddpm_structure_ddim_refine | `ddpm_structure_ddim_refine` (DDPM warm-up + DDIM) or `just_ddim` |
| `--ddpm_ratio` | 0.03 | Fraction of the schedule handled by DDPM (switch timestep = 1000·(1−ratio)) |
| `--ddmp_steps` | 30 | DDPM steps (flag keeps the original spelling) |
| `--ddim_steps` | 100 | DDIM steps |
| `--text_consistency_strength` | 0.8 | Strength of cross-attention text consistency |
| `--text_consistency_layers` | [] | Layers with text consistency (cross-attn, `attn2`) |
| `--ddpm_causal_layers` | [] | Layers with causal alignment during the DDPM stage (self-attn, `attn1`) |
| `--ddim_causal_layers` | [] | Layers with causal alignment during the DDIM stage |
| `--causal_reference_frame_idx` | 0 | Reference frame for causal K/V broadcast |
| `--out` | auto | Output video path |
| `--save_frames` | off | Also save individual frames |
| `--enable_memory_opt` | off | VAE slicing/tiling (for low VRAM) |

Available layers: `down_blocks.0`, `down_blocks.1`, `down_blocks.2`, `mid_block`, `up_blocks.0`, `up_blocks.1`, `up_blocks.2`.

Example with LLM semantic points (9 frames → 9 values):

```bash
python main.py \
  --start_prompt "A young toddler sitting on a wooden floor." \
  --end_prompt "The toddler standing up and taking first steps." \
  --video_length 9 \
  --semantic_points 0.0 0.12 0.30 0.55 0.80 0.95 1.0 1.0 1.0 \
  --power_fast_exponent 0.3
```

## Repository structure

```
main.py                     # CLI entry point (the only runnable script)
model/
  __init__.py               # Package init (re-exports Model, FuVideoPipeline)
  model.py                  # Model wrapper: loading, text2video interface
  text_to_video_pipeline.py # FuVideoPipeline: mixed sampling, embedding
                            #   interpolation, noise-space frame interpolation
  utils.py                  # Attention processors + video I/O utilities
llm_prompt_designer.md      # Design doc for the LLM that produces semantic_points
system_prompt_for_paper_en.txt  # The LLM system prompt (English)
```

## Requirements

Python 3.10, CUDA 11.8+, ~16 GB VRAM for 1024×1024 × 9 frames (use `--enable_memory_opt` for less). Main dependencies: `torch≥2.1`, `diffusers≥0.25`, `transformers≥4.37`.

## License & credit

Code is released under the [CreativeML Open RAIL-M license](LICENSE). This work builds on [Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://arxiv.org/abs/2303.13439) (Khachatryan et al., 2023) — please cite it if you use this repository:

```bibtex
@article{text2video-zero,
    title={Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators},
    author={Khachatryan, Levon and Movsisyan, Andranik and Tadevosyan, Vahram and Henschel, Roberto and Wang, Zhangyang and Navasardyan, Shant and Shi, Humphrey},
    journal={arXiv preprint arXiv:2303.13439},
    year={2023}
}
```
