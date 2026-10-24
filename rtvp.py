"""
FossielMiniMaxRefToVid
Classic ComfyUI version of MiniMaxH3ReferenceToVideo
with permanent optional inputs (no Autogrow).

Limits (official + original node):
- 9 reference images
- 3 reference videos
- 3 paired video soundtracks
- 3 standalone audios
"""

import math
import torch
import torchaudio

import nodes
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE


# ------------------------------------------------------------------
# Constants & helpers (copied / adapted from original)
# ------------------------------------------------------------------
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, z.shape[-1]


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


# ------------------------------------------------------------------
# The node
# ------------------------------------------------------------------
class FossielMiniMaxRefToVid:
    @classmethod
    def INPUT_TYPES(s):
        optional = {
            "vae": ("VAE",),
            "audio_vae": ("VAE",),
        }

        # 9 permanent image slots
        for i in range(1, 10):
            optional[f"ref_image_{i}"] = ("IMAGE",)

        # 3 video slots + 3 paired audio slots
        for i in range(1, 4):
            optional[f"ref_video_{i}"] = ("IMAGE",)
            optional[f"ref_video_audio_{i}"] = ("AUDIO",)

        # 3 standalone audio slots
        for i in range(1, 4):
            optional[f"ref_audio_{i}"] = ("AUDIO",)

        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {
                    "default": 124, "min": 5, "max": 3600, "step": 17,
                    "tooltip": "Frame count at 24 fps (124 ≈ 5s). Snapped to 17k+5 grid."
                }),
                "ref_image_size": (["match", "max"], {
                    "default": "match",
                    "tooltip": "'match' = scale to generation area (down only). 'max' = 2048 short edge (better identity, slower)."
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "execute"
    CATEGORY = "Fossiel/MiniMax"
    DESCRIPTION = "MiniMax H3 Reference-to-Video with fixed optional inputs (no Autogrow). Max 9 images + 3 videos + 3 video-audios + 3 standalone audios."

    def execute(self, clip, prompt, width, height, length, ref_image_size="match",
                vae=None, audio_vae=None,
                # images 1-9
                ref_image_1=None, ref_image_2=None, ref_image_3=None,
                ref_image_4=None, ref_image_5=None, ref_image_6=None,
                ref_image_7=None, ref_image_8=None, ref_image_9=None,
                # videos + paired audios
                ref_video_1=None, ref_video_2=None, ref_video_3=None,
                ref_video_audio_1=None, ref_video_audio_2=None, ref_video_audio_3=None,
                # standalone audios
                ref_audio_1=None, ref_audio_2=None, ref_audio_3=None):

        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # for tokenizer (Qwen)
        ref_blocks = []  # for DiT

        # ----------------------------------------------------------
        # 1. Reference images (order 1 → 9)
        # ----------------------------------------------------------
        images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
                  ref_image_6, ref_image_7, ref_image_8, ref_image_9]

        for img in images:
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            if vae is not None:
                z = vae.encode(resized)
                ref_blocks.append({
                    "kind": "image",
                    "latent_h": th // 16,
                    "latent_w": tw // 16,
                    "latent": z
                })

        # ----------------------------------------------------------
        # 2. Reference videos + their paired soundtracks
        # ----------------------------------------------------------
        videos = [ref_video_1, ref_video_2, ref_video_3]
        video_audios = [ref_video_audio_1, ref_video_audio_2, ref_video_audio_3]

        for i, video_frames in enumerate(videos):
            if video_frames is None:
                continue

            soundtrack = video_audios[i]

            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]

            # soundtrack gets its own <Audio j> label BEFORE the video
            if soundtrack is not None:
                ref_items.append({"type": "audio"})

            # Qwen sees the video at 2 fps
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({
                "type": "video",
                "data": qwen_frames,
                "timestamps": [j / 2.0 for j in range(len(sample_idx))]
            })

            if vae is None:
                continue

            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None and audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)

            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": z.shape[2],
                "latent_h": ch // 16,
                "latent_w": cw // 16,
                "ref_audio_t": ref_audio_t,
                "latent": z,
                "audio_latent": audio_latent
            })

        # ----------------------------------------------------------
        # 3. Standalone reference audios
        # ----------------------------------------------------------
        standalone_audios = [ref_audio_1, ref_audio_2, ref_audio_3]
        for audio in standalone_audios:
            if audio is None:
                continue
            ref_items.append({"type": "audio"})
            if audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
                ref_blocks.append({
                    "kind": "audio",
                    "ref_audio_t": ref_audio_t,
                    "audio_latent": audio_latent
                })

        # ----------------------------------------------------------
        # Tokenize & condition
        # ----------------------------------------------------------
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        return (cond, latent)



