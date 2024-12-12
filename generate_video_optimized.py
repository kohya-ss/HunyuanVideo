import argparse
from datetime import datetime
from pathlib import Path
import random
import sys
import os
import time
from typing import Optional, Union

import torch
from loguru import logger
import accelerate
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from hyvideo.modules.models import HYVideoDiffusionTransformer


# dirty hack to correctly work `model_base` argument
# set MODEL_BASE environment variable based on args.model_base
sys_args = sys.argv[1:]
for i, arg in enumerate(sys_args):
    if arg == "--model-base":
        os.environ["MODEL_BASE"] = sys_args[i + 1]
        logger.info(f"Set MODEL_BASE to {sys_args[i + 1]}")
        break


import hyvideo.config as config
from hyvideo.constants import PROMPT_TEMPLATE, PRECISION_TO_TYPE
from hyvideo.text_encoder import TextEncoder
from hyvideo.vae import load_vae
from hyvideo.modules import load_model
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.utils.file_utils import save_videos_grid


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

    parser = config.add_network_args(parser)
    parser = config.add_extra_models_args(parser)
    parser = config.add_denoise_schedule_args(parser)
    parser = config.add_inference_args(parser)
    parser = config.add_parallel_args(parser)

    # add arguments for this script
    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model, --precision is still used")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument("--attn-mode", type=str, default="flash", help="attention mode for transformer")
    parser.add_argument("--vae-chunk-size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument("--blocks-to-swap", type=int, default=None, help="number of blocks to swap in the model")
    parser.add_argument("--img-in-txt-in-offloading", action="store_true", help="offload img_in and txt_in to cpu")
    parser.add_argument("--output-type", type=str, default="video", help="output type: video, latent or both")
    parser.add_argument("--latent-path", type=str, default=None, help="path to latent for decode. no inference")

    args = parser.parse_args()
    args = config.sanity_check_args(args)

    # extra checks
    assert args.denoise_type == "flow", "only flow denoising is supported in this script"
    assert args.cfg_scale == 1.0, "classifier free guidance is not supported in this script"
    assert args.text_encoder_2 is not None, "text_encoder_2 is required for this script"
    assert args.latent_path is None or args.output_type == "video", "latent-path is only supported with output-type=video"

    # update dit_weight based on model_base if not exists
    if args.dit_weight is not None:
        dit_weight = Path(args.dit_weight)
        if not dit_weight.exists():
            model_base = Path(os.environ.get("MODEL_BASE", args.model_base))
            dit_weight = model_base / dit_weight
            if not dit_weight.exists():
                # remove redundant "ckpts" from path
                dit_weight = model_base.parent / Path(args.dit_weight)
            if dit_weight.exists():
                args.dit_weight = str(dit_weight)
                logger.info(f"Updated dit_weight to {dit_weight}")

    return args


def check_inputs(args):
    height = args.video_size[0]
    width = args.video_size[1]
    video_length = args.video_length
    vae_ver = args.vae

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    if video_length is not None:
        if "884" in vae_ver:
            if video_length != 1 and (video_length - 1) % 4 != 0:
                raise ValueError(f"`video_length` has to be 1 or a multiple of 4 but is {video_length}.")
        elif "888" in vae_ver:
            if video_length != 1 and (video_length - 1) % 8 != 0:
                raise ValueError(f"`video_length` has to be 1 or a multiple of 8 but is {video_length}.")

    return height, width, video_length


def clean_memory_on_device(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "cpu":
        pass
    elif device.type == "mps":  # not tested
        torch.mps.empty_cache()


# region Encoding prompt


def encode_prompt(prompt: Union[str, list[str]], device: torch.device, num_videos_per_prompt: int, text_encoder: TextEncoder):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_videos_per_prompt (`int`):
            number of videos that should be generated per prompt
        text_encoder (TextEncoder):
            text encoder to be used for encoding the prompt
    """
    # LoRA and Textual Inversion are not supported in this script
    # negative prompt and prompt embedding are not supported in this script
    # clip_skip is not supported in this script because it is not used in the original script
    data_type = "video"  # video only, image is not supported

    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

    with torch.no_grad():
        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, device=device)
    prompt_embeds = prompt_outputs.hidden_state

    attention_mask = prompt_outputs.attention_mask
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        bs_embed, seq_len = attention_mask.shape
        attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
        attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

    prompt_embeds_dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    if prompt_embeds.ndim == 2:
        bs_embed, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
    else:
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, attention_mask


def encode_input_prompt(prompt, args, device):
    if args.prompt_template_video is not None:
        crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
    elif args.prompt_template is not None:
        crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
    else:
        crop_start = 0
    max_length = args.text_len + crop_start

    # prompt_template
    prompt_template = PROMPT_TEMPLATE[args.prompt_template] if args.prompt_template is not None else None

    # prompt_template_video
    prompt_template_video = PROMPT_TEMPLATE[args.prompt_template_video] if args.prompt_template_video is not None else None

    # load text encoders
    logger.info(f"loading text encoder: {args.text_encoder}")
    text_encoder = TextEncoder(
        text_encoder_type=args.text_encoder,
        max_length=max_length,
        text_encoder_precision=args.text_encoder_precision,
        tokenizer_type=args.tokenizer,
        prompt_template=prompt_template,
        prompt_template_video=prompt_template_video,
        hidden_state_skip_layer=args.hidden_state_skip_layer,
        apply_final_norm=args.apply_final_norm,
        reproduce=args.reproduce,
        logger=logger,
        device=device,  # if not args.use_cpu_offload else "cpu",
    )
    text_encoder.eval()

    logger.info(f"loading text encoder 2: {args.text_encoder_2}")
    text_encoder_2 = TextEncoder(
        text_encoder_type=args.text_encoder_2,
        max_length=args.text_len_2,
        text_encoder_precision=args.text_encoder_precision_2,
        tokenizer_type=args.tokenizer_2,
        reproduce=args.reproduce,
        logger=logger,
        device=device,  # if not args.use_cpu_offload else "cpu",
    )
    text_encoder_2.eval()

    # encode prompt
    logger.info(f"Encoding prompt with text encoder 1")
    prompt_embeds, prompt_mask = encode_prompt(prompt, device, args.num_videos, text_encoder)
    logger.info(f"Encoding prompt with text encoder 2")
    prompt_embeds_2, prompt_mask_2 = encode_prompt(prompt, device, args.num_videos, text_encoder_2)

    prompt_embeds = prompt_embeds.to("cpu")
    prompt_mask = prompt_mask.to("cpu")
    prompt_embeds_2 = prompt_embeds_2.to("cpu")
    prompt_mask_2 = prompt_mask_2.to("cpu")

    text_encoder = None
    text_encoder_2 = None
    clean_memory_on_device(device)

    return prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2


# endregion


# region transformer
def load_state_dict(args, model, pretrained_model_path):
    load_key = args.load_key
    dit_weight = Path(args.dit_weight)

    if dit_weight is None:
        model_dir = pretrained_model_path / f"t2v_{args.model_resolution}"
        files = list(model_dir.glob("*.pt"))
        if len(files) == 0:
            raise ValueError(f"No model weights found in {model_dir}")
        if str(files[0]).startswith("pytorch_model_"):
            model_path = dit_weight / f"pytorch_model_{load_key}.pt"
            bare_model = True
        elif any(str(f).endswith("_model_states.pt") for f in files):
            files = [f for f in files if str(f).endswith("_model_states.pt")]
            model_path = files[0]
            if len(files) > 1:
                logger.warning(f"Multiple model weights found in {dit_weight}, using {model_path}")
            bare_model = False
        else:
            raise ValueError(
                f"Invalid model path: {dit_weight} with unrecognized weight format: "
                f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                f"specific weight file, please provide the full path to the file."
            )
    else:
        if dit_weight.is_dir():
            files = list(dit_weight.glob("*.pt"))
            if len(files) == 0:
                raise ValueError(f"No model weights found in {dit_weight}")
            if str(files[0]).startswith("pytorch_model_"):
                model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                bare_model = True
            elif any(str(f).endswith("_model_states.pt") for f in files):
                files = [f for f in files if str(f).endswith("_model_states.pt")]
                model_path = files[0]
                if len(files) > 1:
                    logger.warning(f"Multiple model weights found in {dit_weight}, using {model_path}")
                bare_model = False
            else:
                raise ValueError(
                    f"Invalid model path: {dit_weight} with unrecognized weight format: "
                    f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                    f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                    f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                    f"specific weight file, please provide the full path to the file."
                )
        elif dit_weight.is_file():
            model_path = dit_weight
            bare_model = "unknown"
        else:
            raise ValueError(f"Invalid model path: {dit_weight}")

    if not model_path.exists():
        raise ValueError(f"model_path not exists: {model_path}")
    logger.info(f"Loading torch model {model_path}...")
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
        bare_model = False
    if bare_model is False:
        if load_key in state_dict:
            state_dict = state_dict[load_key]
        else:
            raise KeyError(
                f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                f"are: {list(state_dict.keys())}."
            )
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model


def load_transformer(args, pretrained_model_path, device, dtype) -> HYVideoDiffusionTransformer:
    # =========================== Build main model ===========================
    logger.info("Building model...")
    factor_kwargs = {"device": device, "dtype": dtype, "attn_mode": args.attn_mode}
    in_channels = args.latent_channels
    out_channels = args.latent_channels

    with accelerate.init_empty_weights():
        transformer = load_model(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs,
        )
    transformer = load_state_dict(args, transformer, pretrained_model_path)
    logger.info(f"Moving and casting model to {device} and {dtype}")
    transformer.to(device=device, dtype=dtype)
    transformer.eval()

    return transformer


def get_rotary_pos_embed(args, model, video_length, height, width):
    target_ndim = 3
    ndim = 5 - 2
    # 884
    if "884" in args.vae:
        latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
    elif "888" in args.vae:
        latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
    else:
        latents_size = [video_length, height // 8, width // 8]

    if isinstance(model.patch_size, int):
        assert all(s % model.patch_size == 0 for s in latents_size), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // model.patch_size for s in latents_size]
    elif isinstance(model.patch_size, list):
        assert all(s % model.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // model.patch_size[idx] for idx, s in enumerate(latents_size)]

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = model.hidden_size // model.heads_num
    rope_dim_list = model.rope_dim_list
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=args.rope_theta,
        use_real=True,
        theta_rescale_factor=1,
    )
    return freqs_cos, freqs_sin


# endregion


def decode_latents(args, latents, device):
    vae, _, s_ratio, t_ratio = load_vae(args.vae, args.vae_precision, logger=logger, device=device)
    # vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

    # set chunk_size to CausalConv3d recursively
    chunk_size = args.vae_chunk_size
    if chunk_size is not None:

        def set_chunk_size(module):
            if hasattr(module, "chunk_size"):
                module.chunk_size = chunk_size

        vae.apply(set_chunk_size)
        logger.info(f"Set chunk_size to {chunk_size} for CausalConv3d")

    expand_temporal_dim = False
    if len(latents.shape) == 4:
        latents = latents.unsqueeze(2)
        expand_temporal_dim = True
    elif len(latents.shape) == 5:
        pass
    else:
        raise ValueError(f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}.")

    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    else:
        latents = latents / vae.config.scaling_factor

    with torch.no_grad():
        latents = latents.to(device=device, dtype=vae.dtype)
        if args.vae_tiling:
            vae.enable_tiling()
            image = vae.decode(latents, return_dict=False)[0]
        else:
            image = vae.decode(latents, return_dict=False)[0]

    if expand_temporal_dim or image.shape[2] == 1:
        image = image.squeeze(2)

    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().float()

    return image


def main():
    args = parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dit_dtype = PRECISION_TO_TYPE[args.precision]
    dit_weight_dtype = torch.float8_e4m3fn if args.fp8 else dit_dtype
    logger.info(f"Using device: {device}, DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")

    if args.latent_path is not None:
        latents = torch.load(args.latent_path, map_location="cpu")
        logger.info(f"Loaded latent from {args.latent_path}. Shape: {latents.shape}")
        latents = latents.unsqueeze(0)
        seeds = [0]  # dummy seed
    else:
        # prepare accelerator
        mixed_precision = args.precision if args.precision != "fp32" else None
        accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

        # load prompt
        prompt = args.prompt  # TODO load prompts from file
        assert prompt is not None, "prompt is required"

        # check inputs: may be height, width, video_length etc will be changed for each generation in future
        height, width, video_length = check_inputs(args)

        # encode prompt with LLM and Text Encoder
        logger.info(f"Encoding prompt: {prompt}")
        prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2 = encode_input_prompt(prompt, args, device)

        # load DiT model
        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        loading_device = "cpu" if blocks_to_swap > 0 else device

        models_root_path = Path(args.model_base)
        logger.info(f"Loading DiT model from {args.dit_weight} or {models_root_path}")
        transformer = load_transformer(args, models_root_path, loading_device, dit_weight_dtype)

        if blocks_to_swap > 0:
            logger.info(f"Enable swap {blocks_to_swap} blocks to CPU from device: {device}")
            transformer.enable_block_swap(blocks_to_swap, device)
            transformer.move_to_device_except_swap_blocks(device)
            transformer.prepare_block_swap_before_forward()
        if args.img_in_txt_in_offloading:
            logger.info("Enable offloading img_in and txt_in to CPU")
            transformer.enable_img_in_txt_in_offloading()

        # load scheduler
        logger.info(f"Loading scheduler")
        scheduler = FlowMatchDiscreteScheduler(shift=args.flow_shift, reverse=args.flow_reverse, solver=args.flow_solver)

        # Prepare timesteps
        num_inference_steps = args.infer_steps
        scheduler.set_timesteps(num_inference_steps, device=device)  # n_tokens is not used in FlowMatchDiscreteScheduler
        timesteps = scheduler.timesteps

        # Prepare generator
        num_videos_per_prompt = args.num_videos
        seed = args.seed
        if seed is None:
            seeds = [random.randint(0, 1_000_000) for _ in range(num_videos_per_prompt)]
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(num_videos_per_prompt)]
        else:
            raise ValueError(f"Seed must be an integer or None, got {seed}.")
        generator = [torch.Generator(device).manual_seed(seed) for seed in seeds]

        # Prepare latents
        num_channels_latents = transformer.config.in_channels
        vae_scale_factor = 2 ** (4 - 1)  # len(self.vae.config.block_out_channels) == 4

        if "884" in args.vae:
            latent_video_length = (video_length - 1) // 4 + 1
        elif "888" in args.vae:
            latent_video_length = (video_length - 1) // 8 + 1
        else:
            latent_video_length = video_length

        shape = (
            num_videos_per_prompt,
            num_channels_latents,
            latent_video_length,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dit_dtype)
        # FlowMatchDiscreteScheduler does not have init_noise_sigma

        # Denoising loop
        embedded_guidance_scale = args.embedded_cfg_scale
        if embedded_guidance_scale is not None:
            guidance_expand = torch.tensor([embedded_guidance_scale * 1000.0] * latents.shape[0], dtype=torch.float32, device="cpu")
            guidance_expand = guidance_expand.to(device=device, dtype=dit_dtype)
        else:
            guidance_expand = None
        freqs_cos, freqs_sin = get_rotary_pos_embed(args, transformer, video_length, height, width)
        # n_tokens = freqs_cos.shape[0]

        # move and cast all inputs to the correct device and dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=dit_dtype)
        prompt_mask = prompt_mask.to(device=device)
        prompt_embeds_2 = prompt_embeds_2.to(device=device, dtype=dit_dtype)
        prompt_mask_2 = prompt_mask_2.to(device=device)
        freqs_cos = freqs_cos.to(device=device, dtype=dit_dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dit_dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents = scheduler.scale_model_input(latents, t)

                # predict the noise residual
                with torch.no_grad(), accelerator.autocast():
                    noise_pred = transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        latents,  # [1, 16, 33, 24, 42]
                        t.repeat(latents.shape[0]).to(device=device, dtype=dit_dtype),  # [1]
                        text_states=prompt_embeds,  # [1, 256, 4096]
                        text_mask=prompt_mask,  # [1, 256]
                        text_states_2=prompt_embeds_2,  # [1, 768]
                        freqs_cos=freqs_cos,  # [seqlen, head_dim]
                        freqs_sin=freqs_sin,  # [seqlen, head_dim]
                        guidance=guidance_expand,
                        return_dict=True,
                    )["x"]

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        transformer = None
        clean_memory_on_device(device)

    # Save samples
    output_type = args.output_type
    save_path = args.save_path if args.save_path_suffix == "" else f"{args.save_path}_{args.save_path_suffix}"
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    if output_type == "latent" or output_type == "both":
        # save latent
        for i, latent in enumerate(latents):
            latent_path = f"{save_path}/{time_flag}_{i}_{seeds[i]}_latent.pt"
            torch.save(latent, latent_path)
            logger.info(f"Latent save to: {latent_path}")
    if output_type == "video" or output_type == "both":
        # save video
        videos = decode_latents(args, latents, device)
        for i, sample in enumerate(videos):
            sample = sample.unsqueeze(0)
            save_path = f"{save_path}/{time_flag}_{seeds[i]}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f"Sample save to: {save_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
