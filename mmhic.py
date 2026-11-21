import math
import re
import torch
import nodes
import comfy.model_management
import comfy.utils
import node_helpers

CANVAS_MULTIPLE = 32
IMAGE_SIZE = 2048
IMAGE_REF_SHORT_EDGE = 1024
IMAGE_REF_VAE_MAX_SHORT_EDGE = 2048

# ---------------------------------------------------------------------------
# MiniMax shorthand → official tag translation
# ---------------------------------------------------------------------------

_OFFICIAL_TAG = re.compile(
    r"<(?:Subject|Picture|Video|Audio)\s+\d+>",
    re.IGNORECASE,
)

_SUBJECT_PAT = re.compile(
    r"(?<!<)\b(?:Sub(?:ject)?|Char(?:acter)?|Actor|Actress)\s*(\d+)\b(?!>)",
    re.IGNORECASE,
)

_PICTURE_PAT = re.compile(
    r"(?<!<)\b(?:Pic(?:ture)?|Image|Img)\s*(\d+)\b(?!>)",
    re.IGNORECASE,
)

_VIDEO_PAT = re.compile(
    r"(?<!<)\b(?:Vid(?:eo)?)\s*(\d+)\b(?!>)",
    re.IGNORECASE,
)

_AUDIO_PAT = re.compile(
    r"(?<!<)\b(?:Aud(?:io)?)\s*(\d+)\b(?!>)",
    re.IGNORECASE,
)


def minimax_translate_prompt(text: str) -> str:
    if not text:
        return text

    placeholders = []
    def _protect(m):
        placeholders.append(m.group(0))
        return f"\x00MMTAG{len(placeholders)-1}\x00"

    protected = _OFFICIAL_TAG.sub(_protect, text)

    protected = _SUBJECT_PAT.sub(lambda m: f"<Subject {int(m.group(1))}>", protected)
    protected = _PICTURE_PAT.sub(lambda m: f"<Picture {int(m.group(1))}>", protected)
    protected = _VIDEO_PAT.sub(lambda m: f"<Video {int(m.group(1))}>", protected)
    protected = _AUDIO_PAT.sub(lambda m: f"<Audio {int(m.group(1))}>", protected)

    for i, tag in enumerate(placeholders):
        protected = protected.replace(f"\x00MMTAG{i}\x00", tag)

    return protected


def _size_short_edge(width, height, low, high):
    scale = min(max(min(width, height), low), high) / min(width, height)
    return tuple(
        max(CANVAS_MULTIPLE, round(side * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        for side in (width, height)
    )


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _resize_to(image, size):
    if (image.shape[2], image.shape[1]) == size:
        return image[..., :3]
    return _resize(image, size[0], size[1], "disabled")


def _image_refs(images, vae, vision_size, vae_size):
    vision_images = []
    refs = []
    for image in images:
        size = (image.shape[2], image.shape[1])
        vision_images.append(_resize_to(image, vision_size(*size)))
        if vae is not None:
            z = vae.encode(_resize_to(image, vae_size(*size)))
            refs.append({
                "kind": "image",
                "latent_h": z.shape[-2],
                "latent_w": z.shape[-1],
                "latent": z,
            })
    return images, vision_images, refs


class FossielMMHImageConditioning:
    """
    Classic-style MiniMax-H3-Image conditioning.
    Optional Auto-MiniMax_Translation converts shorthand into official tags.
    Final_Prompt outputs the prompt actually sent to the encoder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": nodes.MAX_RESOLUTION,
                    "step": CANVAS_MULTIPLE,
                    "tooltip": "0 uses the width of image1 after its reference scaling, or 2048 without images.",
                }),
                "height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": nodes.MAX_RESOLUTION,
                    "step": CANVAS_MULTIPLE,
                    "tooltip": "0 uses the height of image1 after its reference scaling, or 2048 without images.",
                }),
                "Auto_MiniMax_Translation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Convert Sub1/Subject 1/Char2/Prop3/Outfit4/Object5 → <Subject N> and Pic1/Image2/Img3 → <Picture N> (etc.) before encoding.",
                }),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "The MiniMax-H3-Image VAE. With it the pictures are also anchored as reference latents; without it they only condition the text encoder.",
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "Final_Prompt")
    FUNCTION = "execute"
    CATEGORY = "Fossiel/Conditioning"

    def execute(self, clip, prompt, width, height, Auto_MiniMax_Translation=True,
                vae=None, image1=None, image2=None, image3=None, image4=None, image5=None):

        if Auto_MiniMax_Translation:
            prompt = minimax_translate_prompt(prompt)

        raw = [image1, image2, image3, image4, image5]
        images = [img[:1] for img in raw if img is not None]

        vae_size = lambda w, h: _size_short_edge(
            w, h, IMAGE_REF_SHORT_EDGE, IMAGE_REF_VAE_MAX_SHORT_EDGE
        )
        vision_size = lambda w, h: _size_short_edge(
            w, h, IMAGE_REF_SHORT_EDGE, IMAGE_REF_SHORT_EDGE
        )

        images, vision_images, refs = _image_refs(
            images, vae, vision_size, vae_size
        )

        cond = clip.encode_from_tokens_scheduled(
            clip.tokenize(prompt, images=vision_images, picture_label="Picture %d: ")
        )
        if refs:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": refs})

        if not (width and height):
            size = (
                vae_size(images[0].shape[2], images[0].shape[1])
                if images
                else (IMAGE_SIZE, IMAGE_SIZE)
            )
            width = width or size[0]
            height = height or size[1]

        latent = torch.zeros(
            [1, 24, 1, height // 16, width // 16],
            device=comfy.model_management.intermediate_device(),
        )
        return (cond, {"samples": latent}, prompt)
