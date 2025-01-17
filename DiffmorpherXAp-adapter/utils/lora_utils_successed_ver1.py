from timeit import default_timer as timer
from datetime import timedelta
from PIL import Image
import os
import itertools
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torchvision import transforms
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from packaging import version
from PIL import Image
import tqdm
from typing import Any, Callable, Dict, List, Optional, Union
from transformers import AutoTokenizer, PretrainedConfig
from APadapter.ap_adapter.attention_processor import AttnProcessor2_0,IPAttnProcessor2_0
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
import torchaudio
from audio_encoder.AudioMAE import AudioMAEConditionCTPoolRand, extract_kaldi_fbank_feature
from audioldm.utils import default_audioldm_config
from audioldm.audio import TacotronSTFT, read_wav_file
from audioldm.audio.tools import get_mel_from_wav, _pad_spec, normalize_wav, pad_wav
from transformers import (
    ClapFeatureExtractor,
    ClapModel,
    GPT2Model,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)
from diffusers.utils.torch_utils import randn_tensor
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from torchviz import make_dot
import json
from matplotlib import pyplot as plt
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.17.0")

def wav_to_fbank(
        filename,
        target_length=1024,
        fn_STFT=None,
        augment_data=False,
        mix_data=False,
        snr=None
    ):
    assert fn_STFT is not None
    waveform = read_wav_file(filename, target_length * 160)  # hop size is 160
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )
    fbank = fbank.contiguous()
    log_magnitudes_stft = log_magnitudes_stft.contiguous()
    waveform = waveform.contiguous()
    return fbank, log_magnitudes_stft, waveform

def wav_to_mel(
        original_audio_file_path,
        duration,
        augment_data=False,
        mix_data=False,
        snr=None):
    config=default_audioldm_config()
    
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    mel, _, _ = wav_to_fbank(
        original_audio_file_path,
        target_length=int(duration * 102.4),
        fn_STFT=fn_STFT,
        augment_data=augment_data,
        mix_data=mix_data,
        snr=snr
    )
    mel = mel.unsqueeze(0)
    return mel

def prepare_inputs_for_generation(
    inputs_embeds,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    if past_key_values is not None:
        # only last token for inputs_embeds if past is defined in kwargs
        inputs_embeds = inputs_embeds[:, -1:]
    kwargs["use_cache"] = True
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
    }

def generate_language_model(
        language_model,
        inputs_embeds: torch.Tensor = None,
        max_new_tokens: int = 512,
        **model_kwargs,
    ):
        """

        Generates a sequence of hidden-states from the language model, conditioned on the embedding inputs.

        Parameters:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence used as a prompt for the generation.
            max_new_tokens (`int`):
                Number of new tokens to generate.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the `forward`
                function of the model.

        Return:
            `inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence of generated hidden-states.
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else language_model.config.max_new_tokens
        model_kwargs = language_model._get_initial_cache_position(inputs_embeds, model_kwargs)
        for _ in range(max_new_tokens):
            # prepare model inputs
            model_inputs = prepare_inputs_for_generation(inputs_embeds, **model_kwargs)

            # forward pass to get next hidden states
            output = language_model(**model_inputs, return_dict=True)
            next_hidden_states = output.last_hidden_state

            # Update the model input
            inputs_embeds = torch.cat([inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)

            # Update generated hidden states, model inputs, and length for next step
            model_kwargs = language_model._update_model_kwargs_for_generation(output, model_kwargs)

        return inputs_embeds[:, -max_new_tokens:, :]

def encode_prompt(
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        projection_model,
        language_model,
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        # Define tokenizers and text encoders
        tokenizers = [tokenizer, tokenizer_2]
        text_encoders = [text_encoder, text_encoder_2]

        if prompt_embeds is None:
            prompt_embeds_list = []
            attention_mask_list = []

            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length" if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)) else True,
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                attention_mask = text_inputs.attention_mask
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    # logger.warning(
                    #     f"The following part of your input was truncated because {text_encoder.config.model_type} can "
                    #     f"only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                    # )

                text_input_ids = text_input_ids.to(device)
                attention_mask = attention_mask.to(device)

                if text_encoder.config.model_type == "clap":
                    prompt_embeds = text_encoder.get_text_features(
                        text_input_ids,
                        attention_mask=attention_mask,
                    )
                    # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                    prompt_embeds = prompt_embeds[:, None, :]
                    # make sure that we attend to this single hidden-state
                    attention_mask = attention_mask.new_ones((batch_size, 1))
                else:
                    prompt_embeds = text_encoder(
                        text_input_ids,
                        attention_mask=attention_mask,
                    )
                    prompt_embeds = prompt_embeds[0]

                prompt_embeds_list.append(prompt_embeds)
                attention_mask_list.append(attention_mask) 
            projection_output = projection_model(
                hidden_states=prompt_embeds_list[0],
                hidden_states_1=prompt_embeds_list[1],
                attention_mask=attention_mask_list[0],
                attention_mask_1=attention_mask_list[1],
            )
            projected_prompt_embeds = projection_output.hidden_states
            projected_attention_mask = projection_output.attention_mask

            generated_prompt_embeds = generate_language_model(
                language_model,
                projected_prompt_embeds,
                attention_mask=projected_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        prompt_embeds = prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)
        attention_mask = (
            attention_mask.to(device=device)
            if attention_mask is not None
            else torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)
        )
        generated_prompt_embeds = generated_prompt_embeds.to(dtype=language_model.dtype, device=device)

        bs_embed, seq_len, hidden_size = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len, hidden_size)

        # duplicate attention mask for each generation per prompt
        attention_mask = attention_mask.repeat(1, num_waveforms_per_prompt)
        attention_mask = attention_mask.view(bs_embed * num_waveforms_per_prompt, seq_len)

        bs_embed, seq_len, hidden_size = generated_prompt_embeds.shape
        # duplicate generated embeddings for each generation per prompt, using mps friendly method
        generated_prompt_embeds = generated_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        generated_prompt_embeds = generated_prompt_embeds.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            negative_attention_mask_list = []
            max_length = prompt_embeds.shape[1]
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=tokenizer.model_max_length
                    if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
                    else max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                uncond_input_ids = uncond_input.input_ids.to(device)
                negative_attention_mask = uncond_input.attention_mask.to(device)

                if text_encoder.config.model_type == "clap":
                    negative_prompt_embeds = text_encoder.get_text_features(
                        uncond_input_ids,
                        attention_mask=negative_attention_mask,
                    )
                    # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                    negative_prompt_embeds = negative_prompt_embeds[:, None, :]
                    # make sure that we attend to this single hidden-state
                    negative_attention_mask = negative_attention_mask.new_ones((batch_size, 1))
                else:
                    negative_prompt_embeds = text_encoder(
                        uncond_input_ids,
                        attention_mask=negative_attention_mask,
                    )
                    negative_prompt_embeds = negative_prompt_embeds[0]

                negative_prompt_embeds_list.append(negative_prompt_embeds)
                negative_attention_mask_list.append(negative_attention_mask)

            projection_output = projection_model(
                hidden_states=negative_prompt_embeds_list[0],
                hidden_states_1=negative_prompt_embeds_list[1],
                attention_mask=negative_attention_mask_list[0],
                attention_mask_1=negative_attention_mask_list[1],
            )
            negative_projected_prompt_embeds = projection_output.hidden_states
            negative_projected_attention_mask = projection_output.attention_mask

            negative_generated_prompt_embeds = generate_language_model(
                language_model,
                negative_projected_prompt_embeds,
                attention_mask=negative_projected_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)
            negative_attention_mask = (
                negative_attention_mask.to(device=device)
                if negative_attention_mask is not None
                else torch.ones(negative_prompt_embeds.shape[:2], dtype=torch.long, device=device)
            )
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.to(
                dtype=language_model.dtype, device=device
            )

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_waveforms_per_prompt, seq_len, -1)

            # duplicate unconditional attention mask for each generation per prompt
            negative_attention_mask = negative_attention_mask.repeat(1, num_waveforms_per_prompt)
            negative_attention_mask = negative_attention_mask.view(batch_size * num_waveforms_per_prompt, seq_len)

            # duplicate unconditional generated embeddings for each generation per prompt
            seq_len = negative_generated_prompt_embeds.shape[1]
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.view(
                batch_size * num_waveforms_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            attention_mask = torch.cat([negative_attention_mask, attention_mask])
            generated_prompt_embeds = torch.cat([negative_generated_prompt_embeds, generated_prompt_embeds])
        
        return prompt_embeds, attention_mask, generated_prompt_embeds

def prepare_latents(vae, vocoder, scheduler, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    shape = (
        batch_size,
        num_channels_latents,
        height // vae_scale_factor,
        vocoder.config.model_in_dim // vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * scheduler.init_noise_sigma
    return latents

def plot_loss(loss_history, loss_plot_path, lora_steps):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, lora_steps + 1), loss_history, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()
    # print(f"Loss plot saved to {loss_plot_path}")


# model_path: path of the model
# image: input image, have not been pre-processed
# save_lora_dir: the path to save the lora
# prompt: the user input prompt
# lora_steps: number of lora training step
# lora_lr: learning rate of lora training
# lora_rank: the rank of lora
def train_lora(audio_path ,dtype ,time_pooling ,freq_pooling ,prompt, negative_prompt, guidance_scale, save_lora_dir, tokenizer=None, tokenizer_2=None,
               text_encoder=None, text_encoder_2=None, GPT2=None, projection_model=None, vocoder=None,
               vae=None, unet=None, noise_scheduler=None, lora_steps=200, lora_lr=2e-4, lora_rank=16, weight_name=None, safe_serialization=False, progress=tqdm):
    time_pooling = time_pooling
    freq_pooling = freq_pooling 
    # initialize accelerator
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=1,
    #     mixed_precision='no'
    # )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(0)
    # set device and dtype
    # prepare accelerator
    # unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
    # optimizer = accelerator.prepare_optimizer(optimizer)
    # lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    GPT2.requires_grad_(False)
    projection_model.requires_grad_(False)
    vocoder.requires_grad_(False)
    unet.requires_grad_(False)

    


    for name, param in text_encoder_2.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in GPT2.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in vae.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in vocoder.named_parameters():
        if param.requires_grad:
            print(name)

    unet.to(device)
    vae.to(device)
    text_encoder.to(device)


    # initialize UNet LoRA
    unet_lora_attn_procs = {}
    i = 0 # Counter variable to iterate through the cross-attention dimension array.
    cross = [None, None, 768, 768, 1024, 1024, None, None] # Predefined cross-attention dimensions for different layers.
    do_copy = False
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise NotImplementedError("name must start with up_blocks, mid_blocks, or down_blocks")

        # if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
        #     lora_attn_processor_class = LoRAAttnAddedKVProcessor
        # else:
        #     lora_attn_processor_class = (
        #         LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
        #     )
        
        if cross_attention_dim is None:
            unet_lora_attn_procs[name] = AttnProcessor2_0()
        else:
            cross_attention_dim = cross[i%8]
            i += 1
            if cross_attention_dim == 768:
                unet_lora_attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    name = name,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=8,
                    do_copy = do_copy
                ).to(device, dtype=dtype)
            else:
                unet_lora_attn_procs[name] = AttnProcessor2_0()
    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # Optimizer creation
    params_to_optimize = (unet_lora_layers.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=lora_steps,
        num_cycles=1,
        power=1.0,
    )


    do_classifier_free_guidance = guidance_scale > 1.0
    # initialize text embeddings
    with torch.no_grad():
        prompt_embeds, attention_mask, generated_prompt_embeds = encode_prompt(
            tokenizer,
            tokenizer_2,
            text_encoder,
            text_encoder_2,
            projection_model,
            GPT2,
            prompt,
            device,
            num_waveforms_per_prompt = 1,
            do_classifier_free_guidance= do_classifier_free_guidance,
            negative_prompt = negative_prompt,
        )
    waveform, sr = torchaudio.load(audio_path)
    fbank = torch.zeros((1024, 128))
    ta_kaldi_fbank = extract_kaldi_fbank_feature(waveform, sr, fbank)
    mel_spect_tensor = ta_kaldi_fbank.unsqueeze(0)
    model = AudioMAEConditionCTPoolRand().to(device).to(dtype=dtype)
    model.eval()
    mel_spect_tensor = mel_spect_tensor.to(device, dtype=next(model.parameters()).dtype)
    LOA_embed = model(mel_spect_tensor, time_pool=time_pooling, freq_pool=freq_pooling)
    uncond_LOA_embed = model(torch.zeros_like(mel_spect_tensor), time_pool=time_pooling, freq_pool=freq_pooling)
    LOA_embeds = LOA_embed[0]
    uncond_LOA_embeds = uncond_LOA_embed[0]
    bs_embed, seq_len, _ = LOA_embeds.shape
    num = prompt_embeds.shape[0] // 2
    LOA_embeds = LOA_embeds.view(bs_embed , seq_len, -1)
    LOA_embeds = LOA_embeds.repeat(num, 1, 1)
    uncond_LOA_embeds = uncond_LOA_embeds.view(bs_embed , seq_len, -1)
    uncond_LOA_embeds = uncond_LOA_embeds.repeat(num, 1, 1)
    negative_g, g = generated_prompt_embeds.chunk(2)
    uncond = torch.cat([negative_g, uncond_LOA_embeds], dim=1)
    cond = torch.cat([g, LOA_embeds], dim=1)
    generated_prompt_embeds = torch.cat([uncond, cond], dim=0)
    model_dtype = next(unet.parameters()).dtype
    generated_prompt_embeds = generated_prompt_embeds.to(model_dtype)
    
    loss_history = []
    if not os.path.exists(save_lora_dir):
        os.makedirs(save_lora_dir)
    weight_path = os.path.join(save_lora_dir, weight_name)
    base_name, _ = os.path.splitext(weight_path)
    save_image_path = f"{base_name}.png"
    print(f'Save image path: {save_image_path}')
    mel_spect_tensor = wav_to_mel(audio_path, duration = 10).unsqueeze(0).to(next(vae.parameters()).dtype)
    
    for step in progress.tqdm(range(lora_steps), desc="Training LoRA..."):
        unet.train()
        # with accelerator.accumulate(unet):
        latents_dist = vae.encode(mel_spect_tensor.to(device)).latent_dist
        model_input = torch.cat([latents_dist.sample()] * 2) if do_classifier_free_guidance else latents_dist.sample()
        model_input = model_input * vae.config.scaling_factor
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input).to(model_input.device)
        bsz, channels, height, width = model_input.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()
        # Add noise to the model input according to the noise magnitude at each timestep (this is the forward diffusion process)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
        generated_prompt_embeds = generated_prompt_embeds.to(device)
        prompt_embeds = prompt_embeds.to(device)
        attention_mask = attention_mask.to(device)
        # Predict the noise residual
        model_pred = unet(sample=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=generated_prompt_embeds,
                        encoder_hidden_states_1=prompt_embeds,
                        encoder_attention_mask_1=attention_mask,
                        return_dict=False,
                        )[0]

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        loss = F.mse_loss(model_pred, target, reduction="mean")
        loss_history.append(loss.item())
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        # with open(loss_log_path, "w") as f:
        #     json.dump(loss_history, f)
        
        plot_loss(loss_history, save_image_path, step+1)


    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
        weight_name=weight_name,
        safe_serialization=safe_serialization
    )
    
def load_lora(unet, lora_0, lora_1, alpha):
    attn_procs = unet.attn_processors
    for name, processor in attn_procs.items():
        if hasattr(processor, 'to_v_ip') or hasattr(processor, 'to_k_ip'):
            weight_name_v = name + ".to_v_ip.weight"
            weight_name_k = name + ".to_k_ip.weight"
            if weight_name_v in lora_0 and weight_name_v in lora_1:
                v_weight = (1 - alpha) * lora_0[weight_name_v] + alpha * lora_1[weight_name_v]
                processor.to_v_ip.weight = torch.nn.Parameter(v_weight.half())
            
            if weight_name_k in lora_0 and weight_name_k in lora_1:
                k_weight = (1 - alpha) * lora_0[weight_name_k] + alpha * lora_1[weight_name_k]
                processor.to_k_ip.weight = torch.nn.Parameter(k_weight.half())
    unet.set_attn_processor(attn_procs)
    return unet
