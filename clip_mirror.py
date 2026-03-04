"""
CLIP Mirror - Auto Anti-Prompt via Embedding Negation
Drop into ComfyUI/custom_nodes/ and restart ComfyUI.

Encodes your positive prompt through CLIP, then negates the entire
embedding vector to create a mathematically precise conceptual opposite.
No token lookup, no word bank — pure vector negation in CLIP space.

Outputs both POSITIVE and NEGATIVE conditioning, ready to plug into
any sampler (works great with LockSeed Martin).

The entire technique is: neg = -pos. That's it. Five lines that matter.
"""

import torch


class CLIPMirror:
    """
    Encodes a text prompt and produces both the positive conditioning
    and its exact negation as the negative conditioning.
    
    The negated CLIP vector is the mathematical opposite of every concept
    in your prompt — giving CFG a maximally informative anchor to push
    away from, instead of pushing away from the empty string (nothing).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip":   ("CLIP",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                }),
            },
            "optional": {
                "mirror_strength": ("FLOAT", {
                    "default": -1.0,
                    "min": -5.0,
                    "max": 0.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Negation multiplier. -1.0 = exact mirror. "
                               "-0.5 = half-strength. -2.0 = double push.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POSITIVE", "NEGATIVE",)
    FUNCTION = "mirror"
    CATEGORY = "conditioning"

    def mirror(self, clip, prompt, mirror_strength=-1.0):
        # Encode the positive prompt through CLIP (handles both
        # CLIP-L and CLIP-G for SDXL automatically via ComfyUI)
        tokens = clip.tokenize(prompt)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)

        pos_cond = output.pop("cond")
        pos_pooled = output.pop("pooled_output")

        # --- The entire CLIP Mirror technique ---
        neg_cond = pos_cond * mirror_strength
        neg_pooled = pos_pooled * mirror_strength
        # ----------------------------------------

        # Build conditioning in ComfyUI's format:
        # List of [cond_tensor, dict_with_pooled]
        # Pass through any extra keys (width, height, crop, etc.)
        pos_extra = {"pooled_output": pos_pooled}
        neg_extra = {"pooled_output": neg_pooled}

        # Carry forward any additional metadata from encoding
        for key in output:
            pos_extra[key] = output[key]
            neg_extra[key] = output[key]

        positive = [[pos_cond, pos_extra]]
        negative = [[neg_cond, neg_extra]]

        print(f"[CLIP Mirror] '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
        print(f"[CLIP Mirror] Strength: {mirror_strength} | "
              f"Cond shape: {list(pos_cond.shape)} | "
              f"Pooled shape: {list(pos_pooled.shape)}")

        return (positive, negative,)


class CLIPMirrorAdvanced:
    """
    Advanced version: encode positive prompt normally, then mirror it
    for the negative, but also allow an additional manual negative prompt
    to be COMBINED with the mirror.
    
    Use case: CLIP Mirror handles the broad conceptual opposite automatically,
    while the manual negative targets specific artifacts ("extra fingers").
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip":   ("CLIP",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                }),
                "mirror_strength": ("FLOAT", {
                    "default": -1.0,
                    "min": -5.0,
                    "max": 0.0,
                    "step": 0.1,
                    "display": "number",
                }),
                "manual_blend": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "number",
                    "tooltip": "How much manual negative to blend in. "
                               "0.0 = pure mirror, 1.0 = equal blend.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POSITIVE", "NEGATIVE",)
    FUNCTION = "mirror_advanced"
    CATEGORY = "conditioning"

    def mirror_advanced(self, clip, prompt, negative_prompt="",
                        mirror_strength=-1.0, manual_blend=0.3):
        # Encode positive
        pos_tokens = clip.tokenize(prompt)
        pos_output = clip.encode_from_tokens(
            pos_tokens, return_pooled=True, return_dict=True
        )
        pos_cond = pos_output.pop("cond")
        pos_pooled = pos_output.pop("pooled_output")

        # Mirror
        mirror_cond = pos_cond * mirror_strength
        mirror_pooled = pos_pooled * mirror_strength

        # If manual negative provided, encode and blend
        if negative_prompt.strip():
            neg_tokens = clip.tokenize(negative_prompt)
            neg_output = clip.encode_from_tokens(
                neg_tokens, return_pooled=True, return_dict=True
            )
            manual_cond = neg_output.pop("cond")
            manual_pooled = neg_output.pop("pooled_output")

            # Blend: (1 - blend) * mirror + blend * manual
            b = manual_blend
            neg_cond = (1.0 - b) * mirror_cond + b * manual_cond
            neg_pooled = (1.0 - b) * mirror_pooled + b * manual_pooled

            print(f"[CLIP Mirror+] Blended mirror ({1.0 - b:.0%}) + "
                  f"manual ({b:.0%}): '{negative_prompt[:40]}'")
        else:
            neg_cond = mirror_cond
            neg_pooled = mirror_pooled

        pos_extra = {"pooled_output": pos_pooled}
        neg_extra = {"pooled_output": neg_pooled}

        for key in pos_output:
            pos_extra[key] = pos_output[key]
            neg_extra[key] = pos_output[key]

        positive = [[pos_cond, pos_extra]]
        negative = [[neg_cond, neg_extra]]

        print(f"[CLIP Mirror+] '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
        print(f"[CLIP Mirror+] Strength: {mirror_strength}")

        return (positive, negative,)


NODE_CLASS_MAPPINGS = {
    "CLIPMirror": CLIPMirror,
    "CLIPMirrorAdvanced": CLIPMirrorAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPMirror": "CLIP Mirror",
    "CLIPMirrorAdvanced": "CLIP Mirror (Advanced)",
}
