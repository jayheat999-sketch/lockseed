# LockSeed Martin & CLIP Mirror

**ComfyUI custom nodes for deterministic sampling and auto anti-prompting.**

10 images in 22.6 seconds. Three-word prompt. No distillation. No LoRA. No cherry-picking.

![angry hungry walrus](comfyui05888*.png)
*"angry hungry walrus" — Challenge XL for Winners, 10 steps, CLIP Mirror ON*

---

## What This Does

Two nodes that work together to collapse the diffusion parameter space:

**LockSeed Martin** — A deterministic sampler that anchors all noise injection to a single seed and fully denoises between steps. Instead of each step fighting the last one, every step refines the same latent image from the same coordinate. Steps stop mattering because none are wasted.

**CLIP Mirror** — Encodes your positive prompt through CLIP, then negates the entire embedding vector and feeds it as negative conditioning. The model gets a mathematically precise conceptual opposite to push away from — instead of pushing away from the empty string (nothing). No token lookup, no word bank. Five lines of actual logic.

Together: steps, CFG scale, prompt length, and manual negative prompts all stop being critical variables. Short prompts resolve fully. Low step counts produce coherent output. The negative prompt generates itself.

## Numbers

| Setup | Steps | Time/Image | Notes |
|-------|-------|-----------|-------|
| Challenge XL, ComfyUI, A770 | 10 | **2.26s** | 10 images in 22.6s |
| Base SDXL, HuggingFace Zero GPU | 10 | **1.1s** | "fish" — one word |
| Base SDXL, HuggingFace Zero GPU | 13 | **1.4s** | 3 extra steps = 0.3s more |

## Installation

1. Navigate to your ComfyUI custom nodes folder:
   ```
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repo:
   ```
   git clone https://github.com/YOUR_USERNAME/LockSeed-Martin.git
   ```

   Or just download `lockseed_martin.py` and `clip_mirror.py` and drop them in `custom_nodes/`.

3. Restart ComfyUI.

4. You'll find the nodes under:
   - **sampling** → `LockSeed Martin`
   - **conditioning** → `CLIP Mirror`
   - **conditioning** → `CLIP Mirror (Advanced)`

No dependencies beyond what ComfyUI already has.

## Workflow

The simplest workflow is four nodes:

```
Load Checkpoint
      ↓
  CLIP Mirror  ──→  POSITIVE + NEGATIVE conditioning
      ↓
LockSeed Martin  ──→  LATENT
      ↓
  VAE Decode  ──→  Save Image
```

**CLIP Mirror** replaces both "CLIP Text Encode" nodes. You type your prompt once — it handles positive and negative automatically.

## The Nodes

### LockSeed Martin (Sampler)

A drop-in sampler replacement. Inputs are the same as KSampler:

| Input | Description |
|-------|-------------|
| model | Your loaded checkpoint |
| positive | Positive conditioning (from CLIP Mirror or CLIP Text Encode) |
| negative | Negative conditioning (from CLIP Mirror or CLIP Text Encode) |
| latent_image | Empty latent or img2img input |
| seed | The one seed that anchors everything |
| steps | 10 is plenty. 7 works. It doesn't really matter. |
| cfg | 3.0–7.0 all work. Less critical with CLIP Mirror. |
| scheduler | karras recommended |

**What it does differently:**
- Resets `torch.manual_seed(seed)` before every noise injection — same noise tensor every step
- Treats each denoising output as clean x₀ before adding noise back
- Supports GGUF models (auto-detects patched noise functions)
- Per-step timing with dynamic ETA in the console

### CLIP Mirror (Conditioning)

Takes a CLIP model and your prompt. Outputs both POSITIVE and NEGATIVE conditioning.

| Input | Description |
|-------|-------------|
| clip | CLIP model from your checkpoint |
| prompt | Your text prompt |
| mirror_strength | -1.0 = exact negation (default). -0.5 = gentler. -2.0 = stronger push. |

**What it does:** Encodes your prompt through CLIP, then multiplies the entire embedding by -1. That's the negative conditioning. The model pushes away from the exact conceptual opposite of your prompt instead of pushing away from nothing.

### CLIP Mirror Advanced (Conditioning)

Same as CLIP Mirror but adds a manual negative prompt that blends with the mirror.

| Input | Description |
|-------|-------------|
| clip | CLIP model |
| prompt | Positive prompt |
| negative_prompt | Optional manual negative (e.g., "extra fingers, blurry") |
| mirror_strength | Negation multiplier (-1.0 default) |
| manual_blend | 0.0 = pure mirror, 1.0 = equal blend with manual negative |

Use this when you want the automatic anti-conditioning AND targeted artifact cleanup.

## Why It Works

### LockSeed Martin

Standard samplers let the random number generator advance at each step. Step 5 gets different noise than step 4. This means each step partially undoes the previous step, and the sampler wastes time re-exploring regions it already visited. LockSeed resets the seed before every noise injection, so the noise is identical every time. Combined with sigma-zero cleaning (fully denoising before re-noising), every step refines the same image rather than exploring new territory.

### CLIP Mirror

Classifier-Free Guidance computes: `output = uncond + scale × (cond - uncond)`

With an empty negative prompt, `uncond` is the empty-string embedding — nearly zero. CFG pushes away from nothing, which is why you need high guidance scales (7–15) to get coherent results.

With CLIP Mirror, `uncond` is replaced by the prediction conditioned on **-v** (the negated prompt vector). CFG now pushes away from a maximally informative anchor. Every dimension of your concept is reinforced. The guidance vector doubles in magnitude and perfectly aligns with the desired direction.

This also works on **distilled models** (LCM, Turbo, Lightning) where CFG=1.0 makes the standard negative prompt completely inert. CLIP Mirror bakes the anti-concept into the conditioning itself.

## The Discovery

These techniques were discovered through first-principles experimentation, not by reading papers. Key observations:

1. **Noise is not random dots — it's a superposition of concepts.** When you give SDXL an empty positive prompt and "tie die" as the negative, it generates a monochrome soldier. "Soldier trudging through desert" as negative produces bright flowers. The model is resolving conceptual opposites.

2. **Negative prompts are not artifact filters — they're conceptual boundaries.** They define where in idea-space the model should NOT go, which paradoxically defines where it SHOULD go more precisely than the positive prompt alone.

3. **If the anti-direction is just -v, why translate back to English?** Skip the words. Negate the vector. Done.

## Links

- **Live Demo**: [huggingface.co/spaces/klyfff/lockseed](https://huggingface.co/spaces/klyfff/lockseed)
- **Paper**: "LockSeed Martin & CLIP Mirror: When Steps, CFG, and Prompt Length Stop Mattering" (2026)

## Compatibility

- Works with any SDXL checkpoint (base, fine-tuned, merged)
- Works with LoRA (apply LoRA before the nodes as usual)
- Tested on Intel Arc A770 (16GB) and NVIDIA via HuggingFace Zero GPU
- ComfyUI only (not A1111 — yet)

## License

MIT — do whatever you want with it.
