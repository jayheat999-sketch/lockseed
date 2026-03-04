"""
LockSeed Martin - Zero-Friction Deterministic Sampler
Drop into ComfyUI/custom_nodes/ and restart ComfyUI.

- Stripped down, highly optimized step scheduler.
- Forces x0 after each denoise (no leftover noise).
- Reseeds before each noise injection with a perfectly locked seed.
- Dynamically searches for GGUF patched noise functions.
- Per-step timing with dynamic ETA.
"""

import time
import torch
import comfy.samplers
import comfy.sample
import comfy.model_management


def make_locked_seed_sampler(seed):
    """
    Returns a sampler function that:
    1. Denoises to clean x0 at each step
    2. Dynamically fetches GGUF patched noise (if present)
    3. Re-adds noise with the same locked seed each time
    4. Logs per-step timing with running ETA
    """
    def locked_sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = extra_args or {}
        total_steps = len(sigmas) - 1
        step_times = []
        sample_start = time.time()

        for i in range(total_steps):
            step_start = time.time()
            sigma_cur  = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Denoise: model predicts x0 from noisy x at sigma_cur
            sigma_in = sigma_cur.unsqueeze(0).expand(x.shape[0])
            denoised = model(x, sigma_in, **extra_args)

            if callback is not None:
                callback({"i": i, "x": x, "denoised": denoised, "sigma": sigma_cur})

            if sigma_next > 0:
                # Add noise with locked seed — same coordinate every step
                torch.manual_seed(seed)
                
                # --- GGUF NOISE PATCH ---
                try:
                    noise_fn = getattr(model, "get_make_noise", None)
                    if noise_fn is None and hasattr(model, "inner_model"):
                        noise_fn = getattr(model.inner_model, "get_make_noise", None)
                        
                    if noise_fn is not None:
                        noise = noise_fn()(denoised)
                    else:
                        noise = torch.randn_like(denoised)
                except Exception:
                    noise = torch.randn_like(denoised)
                # ------------------------

                x = denoised + noise * sigma_next
            else:
                x = denoised

            # --- Per-step timing ---
            step_time = time.time() - step_start
            step_times.append(step_time)
            avg = sum(step_times) / len(step_times)
            remaining = total_steps - (i + 1)
            eta = avg * remaining
            elapsed = time.time() - sample_start
            print(f"  Step {i+1}/{total_steps} | {step_time:.3f}s | "
                  f"Avg: {avg:.3f}s/step | ETA: {eta:.1f}s | "
                  f"Elapsed: {elapsed:.1f}s")

        total = time.time() - sample_start
        avg = sum(step_times) / len(step_times) if step_times else 0
        print(f"[LockSeed Martin] Sampling complete: "
              f"{total_steps} steps in {total:.2f}s ({avg:.3f}s/step)")

        return x

    return comfy.samplers.KSAMPLER(locked_sample)


class LockSeedMartinSampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("MODEL",),
                "positive":      ("CONDITIONING",),
                "negative":      ("CONDITIONING",),
                "latent_image":  ("LATENT",),
                "seed":          ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps":         ("INT", {"default": 10, "min": 1, "max": 100, "display": "number"}),
                "cfg":           ("FLOAT", {"default": 3.6, "min": 0.0, "max": 10.0, "step": 0.1, "display": "number"}),
                "sampler_name":  (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler":     (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, positive, negative, latent_image, seed,
               steps, cfg, sampler_name, scheduler):

        t_total_start = time.time()

        device = comfy.model_management.get_torch_device()
        latent = latent_image["samples"].clone()
        latent = comfy.sample.fix_empty_latent_channels(model, latent)

        # Build sigma schedule
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).cpu()

        print(f"{'='*60}")
        print(f"[LockSeed Martin] Seed: {seed} | {steps} steps | "
              f"CFG: {cfg} | Scheduler: {scheduler}")
        print(f"[LockSeed Martin] Latent: {list(latent.shape)} | "
              f"Device: {device}")

        # Custom sampler
        sampler = make_locked_seed_sampler(seed)

        # Initialize Guider
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        # Load Model to GPU
        t_load_start = time.time()
        comfy.model_management.load_models_gpu(
            [model],
            model.memory_required([latent.shape[0] * 2] + list(latent.shape[1:]))
        )
        t_load = time.time() - t_load_start
        print(f"[LockSeed Martin] Model to GPU: {t_load:.2f}s")

        # Set the initial starting noise
        generator = torch.Generator()
        generator.manual_seed(seed)
        initial_noise = torch.randn(latent.shape, generator=generator)

        # Run the sampling loop
        print(f"[LockSeed Martin] Sampling...")
        t_sample_start = time.time()
        result = guider.sample(
            initial_noise,
            latent,
            sampler,
            sigmas.to(device),
            seed=seed,
            disable_pbar=False,
        )
        t_sample = time.time() - t_sample_start

        t_total = time.time() - t_total_start
        print(f"[LockSeed Martin] Total node time: {t_total:.2f}s "
              f"(GPU load: {t_load:.2f}s + Sample: {t_sample:.2f}s)")
        print(f"{'='*60}")

        return ({"samples": result.cpu()},)


NODE_CLASS_MAPPINGS = {
    "LockSeedMartinSampler": LockSeedMartinSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LockSeedMartinSampler": "LockSeed Martin",
}
