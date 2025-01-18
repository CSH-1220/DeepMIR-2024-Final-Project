from audio_encoder.AudioMAE import AudioMAEConditionCTPoolRand, extract_kaldi_fbank_feature
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from APadapter.ap_adapter.attention_processor import AttnProcessor2_0, IPAttnProcessor2_0
import random
import os
import scipy
import safetensors
import numpy as np
import torch
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

from diffusers.loaders import AttnProcsLayers
from diffusers import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_librosa_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .modeling_audioldm2 import AudioLDM2ProjectionModel, AudioLDM2UNet2DConditionModel
from diffusers.loaders import TextualInversionLoaderMixin

from tqdm import tqdm   # for progress bar
from utils.lora_utils_successed_ver1 import train_lora, load_lora, wav_to_mel
from utils.model_utils import slerp, do_replace_attn
from utils.alpha_scheduler import AlphaScheduler
from audioldm.utils import default_audioldm_config
from audioldm.audio import TacotronSTFT, read_wav_file
from audioldm.audio.tools import get_mel_from_wav, _pad_spec, normalize_wav, pad_wav
if is_librosa_available():
    import librosa
import warnings
import matplotlib.pyplot as plt
from .pipeline_audioldm2 import AudioLDM2Pipeline


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def visualize_mel_spectrogram(mel_spect_tensor, output_path=None):
    mel_spect_array = mel_spect_tensor.squeeze().transpose(1, 0).detach().cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(mel_spect_array, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label="Log-Mel Energy")
    plt.title("Mel-Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency Bins")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Mel-spectrogram saved to {output_path}")
    else:
        plt.show()


class StoreProcessor():
    def __init__(self, original_processor, value_dict, name):
        self.original_processor = original_processor
        self.value_dict = value_dict
        self.name = name
        self.value_dict[self.name] = dict()
        self.id = 0

    def __call__(self, attn, hidden_states, *args, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # Is self attention
        if encoder_hidden_states is None:
            # print(f'In StoreProcessor: {self.name} {self.id}')
            self.value_dict[self.name][self.id] = hidden_states.detach()
            self.id += 1
        res = self.original_processor(attn, hidden_states, *args,
                                      encoder_hidden_states=encoder_hidden_states,
                                      attention_mask=attention_mask,
                                      **kwargs)
        return res


class LoadProcessor():
    def __init__(self, original_processor, name, aud1_dict, aud2_dict, alpha, beta=0, lamd=0.6):
        super().__init__()
        self.original_processor = original_processor
        self.name = name
        self.aud1_dict = aud1_dict
        self.aud2_dict = aud2_dict
        self.alpha = alpha
        self.beta = beta
        self.lamd = lamd
        self.id = 0

    def __call__(self, attn, hidden_states, *args, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # Is self attention
        # 判斷是否是自注意力（self-attention）
        if encoder_hidden_states is None:
            # 如果當前索引小於 10 倍的 self.lamd，使用自定義的混合邏輯
            if self.id < 10 * self.lamd:
                map0 = self.aud1_dict[self.name][self.id]
                map1 = self.aud2_dict[self.name][self.id]
                cross_map = self.beta * hidden_states + \
                    (1 - self.beta) * ((1 - self.alpha) * map0 + self.alpha * map1)
                # 調用原始處理器，將 cross_map 作為 encoder_hidden_states 傳入
                res = self.original_processor(attn, hidden_states, *args,
                                              encoder_hidden_states=cross_map,
                                              attention_mask=attention_mask,
                                              **kwargs)
            else:
                # 否則，使用原始的 encoder_hidden_states（可能為 None）
                res = self.original_processor(attn, hidden_states, *args,
                                              encoder_hidden_states=encoder_hidden_states,
                                              attention_mask=attention_mask,
                                              **kwargs)
            
            self.id += 1
            # 如果索引到達 self.aud1_dict[self.name] 的長度，重置索引為 0
            if self.id == len(self.aud1_dict[self.name]):
                self.id = 0
        else:
            # 如果是跨注意力（encoder_hidden_states 不為 None），直接使用原始處理器
            res = self.original_processor(attn, hidden_states, *args,
                                          encoder_hidden_states=encoder_hidden_states,
                                          attention_mask=attention_mask,
                                          **kwargs)

        return res


def prepare_inputs_for_generation(
    inputs_embeds,
    attention_mask=None,
    past_key_values=None,
    **kwargs,):
    if past_key_values is not None:
        # only last token for inputs_embeds if past is defined in kwargs
        inputs_embeds = inputs_embeds[:, -1:]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
    }


class AudioLDM2MorphPipeline(DiffusionPipeline,TextualInversionLoaderMixin):
    r"""
    Pipeline for text-to-audio generation using AudioLDM2.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.ClapModel`]):
            First frozen text-encoder. AudioLDM2 uses the joint audio-text embedding model
            [CLAP](https://huggingface.co/docs/transformers/model_doc/clap#transformers.CLAPTextModelWithProjection),
            specifically the [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) variant. The
            text branch is used to encode the text prompt to a prompt embedding. The full audio-text model is used to
            rank generated waveforms against the text prompt by computing similarity scores.
        text_encoder_2 ([`~transformers.T5EncoderModel`]):
            Second frozen text-encoder. AudioLDM2 uses the encoder of
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) variant.
        projection_model ([`AudioLDM2ProjectionModel`]):
            A trained model used to linearly project the hidden-states from the first and second text encoder models
            and insert learned SOS and EOS token embeddings. The projected hidden-states from the two text encoders are
            concatenated to give the input to the language model.
        language_model ([`~transformers.GPT2Model`]):
            An auto-regressive language model used to generate a sequence of hidden-states conditioned on the projected
            outputs from the two text encoders.
        tokenizer ([`~transformers.RobertaTokenizer`]):
            Tokenizer to tokenize text for the first frozen text-encoder.
        tokenizer_2 ([`~transformers.T5Tokenizer`]):
            Tokenizer to tokenize text for the second frozen text-encoder.
        feature_extractor ([`~transformers.ClapFeatureExtractor`]):
            Feature extractor to pre-process generated audio waveforms to log-mel spectrograms for automatic scoring.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded audio latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded audio latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        vocoder ([`~transformers.SpeechT5HifiGan`]):
            Vocoder of class `SpeechT5HifiGan` to convert the mel-spectrogram latents to the final audio waveform.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: ClapModel,
        text_encoder_2: T5EncoderModel,
        projection_model: AudioLDM2ProjectionModel,
        language_model: GPT2Model,
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
        tokenizer_2: Union[T5Tokenizer, T5TokenizerFast],
        feature_extractor: ClapFeatureExtractor,
        unet: AudioLDM2UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vocoder: SpeechT5HifiGan,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            projection_model=projection_model,
            language_model=language_model,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
            vocoder=vocoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.aud1_dict = dict()
        self.aud2_dict = dict()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = [
            self.text_encoder.text_model,
            self.text_encoder.text_projection,
            self.text_encoder_2,
            self.projection_model,
            self.language_model,
            self.unet,
            self.vae,
            self.vocoder,
            self.text_encoder,
        ]

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def generate_language_model(
        self,
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
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.language_model.config.max_new_tokens
        model_kwargs = self.language_model._get_initial_cache_position(inputs_embeds, model_kwargs)
        for _ in range(max_new_tokens):
            # prepare model inputs
            model_inputs = prepare_inputs_for_generation(inputs_embeds, **model_kwargs)

            # forward pass to get next hidden states
            output = self.language_model(**model_inputs, return_dict=True)

            next_hidden_states = output.last_hidden_state

            # Update the model input
            inputs_embeds = torch.cat([inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)

            # Update generated hidden states, model inputs, and length for next step
            model_kwargs = self.language_model._update_model_kwargs_for_generation(output, model_kwargs)

        return inputs_embeds[:, -max_new_tokens:, :]

    def encode_prompt(
        self,
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
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device (`torch.device`):
                torch device
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed text embeddings from the Flan T5 model. Can be used to easily tweak text inputs, *e.g.*
                prompt weighting. If not provided, text embeddings will be computed from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed negative text embeddings from the Flan T5 model. Can be used to easily tweak text inputs,
                *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
                 *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input
                 argument.
            negative_generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
                inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will
                be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention
                mask will be computed from `negative_prompt` input argument.
            max_new_tokens (`int`, *optional*, defaults to None):
                The number of new tokens to generate with the GPT2 language model.
        Returns:
            prompt_embeds (`torch.FloatTensor`):
                Text embeddings from the Flan T5 model.
            attention_mask (`torch.LongTensor`):
                Attention mask to be applied to the `prompt_embeds`.
            generated_prompt_embeds (`torch.FloatTensor`):
                Text embeddings generated from the GPT2 langauge model.

        Example:

        ```python
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # Get text embedding vectors
        >>> prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
        ...     prompt="Techno music with a strong, upbeat tempo and high melodic riffs",
        ...     device="cuda",
        ...     do_classifier_free_guidance=True,
        ... )

        >>> # Pass text embeddings to pipeline for text-conditional audio generation
        >>> audio = pipe(
        ...     prompt_embeds=prompt_embeds,
        ...     attention_mask=attention_mask,
        ...     generated_prompt_embeds=generated_prompt_embeds,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ... ).audios[0]

        >>> # save generated audio sample
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
        ```"""
        # print("prompt",prompt)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

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
                    logger.warning(
                        f"The following part of your input was truncated because {text_encoder.config.model_type} can "
                        f"only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                    )

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

            projection_output = self.projection_model(
                hidden_states=prompt_embeds_list[0],
                hidden_states_1=prompt_embeds_list[1],
                attention_mask=attention_mask_list[0],
                attention_mask_1=attention_mask_list[1],
            )
            projected_prompt_embeds = projection_output.hidden_states
            projected_attention_mask = projection_output.attention_mask

            generated_prompt_embeds = self.generate_language_model(
                projected_prompt_embeds,
                attention_mask=projected_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        attention_mask = (
            attention_mask.to(device=device)
            if attention_mask is not None
            else torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)
        )
        generated_prompt_embeds = generated_prompt_embeds.to(dtype=self.language_model.dtype, device=device)

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

            projection_output = self.projection_model(
                hidden_states=negative_prompt_embeds_list[0],
                hidden_states_1=negative_prompt_embeds_list[1],
                attention_mask=negative_attention_mask_list[0],
                attention_mask_1=negative_attention_mask_list[1],
            )
            negative_projected_prompt_embeds = projection_output.hidden_states
            negative_projected_attention_mask = projection_output.attention_mask

            negative_generated_prompt_embeds = self.generate_language_model(
                negative_projected_prompt_embeds,
                attention_mask=negative_projected_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_attention_mask = (
                negative_attention_mask.to(device=device)
                if negative_attention_mask is not None
                else torch.ones(negative_prompt_embeds.shape[:2], dtype=torch.long, device=device)
            )
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.to(
                dtype=self.language_model.dtype, device=device
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

    # Copied from diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline.mel_spectrogram_to_waveform
    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    def score_waveforms(self, text, audio, num_waveforms_per_prompt, device, dtype):
        if not is_librosa_available():
            logger.info(
                "Automatic scoring of the generated audio waveforms against the input prompt text requires the "
                "`librosa` package to resample the generated waveforms. Returning the audios in the order they were "
                "generated. To enable automatic scoring, install `librosa` with: `pip install librosa`."
            )
            return audio
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        resampled_audio = librosa.resample(
            audio.numpy(), orig_sr=self.vocoder.config.sampling_rate, target_sr=self.feature_extractor.sampling_rate
        )
        inputs["input_features"] = self.feature_extractor(
            list(resampled_audio), return_tensors="pt", sampling_rate=self.feature_extractor.sampling_rate
        ).input_features.type(dtype)
        inputs = inputs.to(device)

        # compute the audio-text similarity score using the CLAP model
        logits_per_text = self.text_encoder(**inputs).logits_per_text
        # sort by the highest matching generations per prompt
        indices = torch.argsort(logits_per_text, dim=1, descending=True)[:, :num_waveforms_per_prompt]
        audio = torch.index_select(audio, 0, indices.reshape(-1).cpu())
        return audio

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        generated_prompt_embeds=None,
        negative_generated_prompt_embeds=None,
        attention_mask=None,
        negative_attention_mask=None,):
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and (prompt_embeds is None or generated_prompt_embeds is None):
            raise ValueError(
                "Provide either `prompt`, or `prompt_embeds` and `generated_prompt_embeds`. Cannot leave "
                "`prompt` undefined without specifying both `prompt_embeds` and `generated_prompt_embeds`."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_embeds is not None and negative_generated_prompt_embeds is None:
            raise ValueError(
                "Cannot forward `negative_prompt_embeds` without `negative_generated_prompt_embeds`. Ensure that"
                "both arguments are specified"
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if attention_mask is not None and attention_mask.shape != prompt_embeds.shape[:2]:
                raise ValueError(
                    "`attention_mask should have the same batch size and sequence length as `prompt_embeds`, but got:"
                    f"`attention_mask: {attention_mask.shape} != `prompt_embeds` {prompt_embeds.shape}"
                )

        if generated_prompt_embeds is not None and negative_generated_prompt_embeds is not None:
            if generated_prompt_embeds.shape != negative_generated_prompt_embeds.shape:
                raise ValueError(
                    "`generated_prompt_embeds` and `negative_generated_prompt_embeds` must have the same shape when "
                    f"passed directly, but got: `generated_prompt_embeds` {generated_prompt_embeds.shape} != "
                    f"`negative_generated_prompt_embeds` {negative_generated_prompt_embeds.shape}."
                )
            if (
                negative_attention_mask is not None
                and negative_attention_mask.shape != negative_prompt_embeds.shape[:2]
            ):
                raise ValueError(
                    "`attention_mask should have the same batch size and sequence length as `prompt_embeds`, but got:"
                    f"`attention_mask: {negative_attention_mask.shape} != `prompt_embeds` {negative_prompt_embeds.shape}"
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents with width->self.vocoder.config.model_in_dim
    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            self.vocoder.config.model_in_dim // self.vae_scale_factor,
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
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def pre_check(self, audio_length_in_s, prompt, callback_steps, negative_prompt):
        """
            Step 0: Convert audio input length from seconds to spectrogram height
            Step 1. Check inputs. Raise error if not correct
        """
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt,
        )

        return height, original_waveform_length

    def encode_prompt_for_2_sources(self, prompt_1, prompt_2, negative_prompt_1, negative_prompt_2, max_new_tokens, device, num_waveforms_per_prompt, do_classifier_free_guidance):
        prompt_embeds_1, attention_mask_1, generated_prompt_embeds_1 = self.encode_prompt(
            prompt_1,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt_1,
            max_new_tokens=max_new_tokens,
        )

        prompt_embeds_2, attention_mask_2, generated_prompt_embeds_2 = self.encode_prompt(
            prompt_2,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt_2,
            max_new_tokens=max_new_tokens,
        )
        return [prompt_embeds_1, attention_mask_1, generated_prompt_embeds_1], [prompt_embeds_2, attention_mask_2, generated_prompt_embeds_2]

    def process_encoded_prompt(self, encoded_prompt, audio_file, time_pooling, freq_pooling):
        prompt_embeds, attention_mask, generated_prompt_embeds = encoded_prompt
        waveform, sr = torchaudio.load(audio_file)
        fbank = torch.zeros((1024, 128))
        ta_kaldi_fbank = extract_kaldi_fbank_feature(waveform, sr, fbank)
        # print("ta_kaldi_fbank.shape",ta_kaldi_fbank.shape)
        mel_spect_tensor = ta_kaldi_fbank.unsqueeze(0)
        model = AudioMAEConditionCTPoolRand().cuda()
        model.eval()
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
        model_dtype = next(self.unet.parameters()).dtype
        # Convert your tensor to the same dtype as the model
        generated_prompt_embeds = generated_prompt_embeds.to(model_dtype)

        return prompt_embeds, attention_mask, generated_prompt_embeds

    def init_trained_pipeline(self, model_path, device, dtype, ap_scale, text_ap_scale):
        pipeline_trained = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-large", torch_dtype=dtype).to(device)
        layer_num = 0
        cross = [None, None, 768, 768, 1024, 1024, None, None]
        unet = pipeline_trained.unet
        attn_procs = {}
        for name in  unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
            else:
                cross_attention_dim = cross[layer_num % 8]
                layer_num += 1
                if cross_attention_dim == 768:
                    attn_procs[name] = IPAttnProcessor2_0(
                        hidden_size=hidden_size,
                        name=name,
                        flag='trained',
                        cross_attention_dim=cross_attention_dim,
                        text_scale=text_ap_scale,
                        scale=ap_scale,
                        num_tokens=8,
                        do_copy=False
                    ).to(device, dtype=dtype)
                else:
                    attn_procs[name] = AttnProcessor2_0()

        state_dict = torch.load(model_path, map_location=device)
        for name, processor in attn_procs.items():
            if hasattr(processor, 'to_v_ip') or hasattr(processor, 'to_k_ip'):
                weight_name_v = name + ".to_v_ip.weight"
                weight_name_k = name + ".to_k_ip.weight"
                if dtype == torch.float32:
                    processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].float())
                    processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].float())
                elif dtype == torch.float16:
                    processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].half())
                    processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].half())
        unet.set_attn_processor(attn_procs)
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return unet(*args, **kwargs)

        unet = _Wrapper(unet.attn_processors)

        return pipeline_trained

    @torch.no_grad()
    def aud2latent(self, audio_path, audio_length_in_s):
        DEVICE = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # waveform, sr = torchaudio.load(audio_path)
        # fbank = torch.zeros((height, 64))
        # ta_kaldi_fbank = extract_kaldi_fbank_feature(waveform, sr, fbank, num_mels=64)
        # mel_spect_tensor = ta_kaldi_fbank.unsqueeze(0).unsqueeze(0)

        mel_spect_tensor = wav_to_mel(audio_path, duration=audio_length_in_s).unsqueeze(0)
        output_path = audio_path.replace('.wav', '_fbank.png')
        visualize_mel_spectrogram(mel_spect_tensor, output_path)
        mel_spect_tensor = mel_spect_tensor.to(next(self.vae.parameters()).dtype)
        # print(f'mel_spect_tensor dtype: {mel_spect_tensor.dtype}')
        # print(f'self.vae dtype: {next(self.vae.parameters()).dtype}')
        latents = self.vae.encode(mel_spect_tensor.to(DEVICE))['latent_dist'].mean
        return latents
    
    @torch.no_grad()
    def ddim_inversion(self, start_latents, prompt_embeds, attention_mask, generated_prompt_embeds, guidance_scale,num_inference_steps): 
        start_step = 0
        # print(f"Scheduler timesteps: {self.scheduler.timesteps}")
        num_inference_steps = min(num_inference_steps, int(max(self.scheduler.timesteps)))
        device = start_latents.device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        start_latents *= self.scheduler.init_noise_sigma
        latents = start_latents.clone()
        for i in tqdm(range(start_step, num_inference_steps)):
            t = self.scheduler.timesteps[i]
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1. else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=generated_prompt_embeds, encoder_hidden_states_1=prompt_embeds, encoder_attention_mask_1=attention_mask).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents
    
    def generate_morphing_prompt(self, prompt_1, prompt_2, alpha):
        closer_prompt = prompt_1 if alpha <= 0.5 else prompt_2
        prompt = (
            f"Jazz style music"
        )
        return prompt

    @torch.no_grad()
    def cal_latent(self,audio_length_in_s,time_pooling, freq_pooling,num_inference_steps, guidance_scale, aud_noise_1, aud_noise_2, prompt_1, prompt_2, 
                   prompt_embeds_1, attention_mask_1, generated_prompt_embeds_1, prompt_embeds_2, attention_mask_2, generated_prompt_embeds_2,
                   alpha, original_processor,attn_processor_dict, use_morph_prompt, morphing_with_lora):
        num_inference_steps = min(num_inference_steps, int(max(self.pipeline_trained.scheduler.timesteps)))
        latents = slerp(aud_noise_1, aud_noise_2, alpha, self.use_adain)
        # vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate
        # height = int(audio_length_in_s / vocoder_upsample_factor)
        # num_channels_latents = self.unet.config.in_channels
        # device = latents.device
        # generator = None
        # latents = self.prepare_latents(
        #     1,
        #     num_channels_latents,
        #     height,
        #     prompt_embeds_1.dtype,
        #     device,
        #     generator,
        #     # latents,
        # )
        if not use_morph_prompt:
            print("Not using morphing prompt")
            max_length = max(prompt_embeds_1.shape[1], prompt_embeds_2.shape[1])
            if prompt_embeds_1.shape[1] < max_length:
                pad_size = max_length - prompt_embeds_1.shape[1]
                padding = torch.zeros(
                    (prompt_embeds_1.shape[0], pad_size, prompt_embeds_1.shape[2]),
                    device=prompt_embeds_1.device,
                    dtype=prompt_embeds_1.dtype
                )
                prompt_embeds_1 = torch.cat([prompt_embeds_1, padding], dim=1)
            
            if prompt_embeds_2.shape[1] < max_length:
                pad_size = max_length - prompt_embeds_2.shape[1]
                padding = torch.zeros(
                    (prompt_embeds_2.shape[0], pad_size, prompt_embeds_2.shape[2]),
                    device=prompt_embeds_2.device,
                    dtype=prompt_embeds_2.dtype
                )
                prompt_embeds_2 = torch.cat([prompt_embeds_2, padding], dim=1)
            
            if attention_mask_1.shape[1] < max_length:
                pad_size = max_length - attention_mask_1.shape[1]
                padding = torch.zeros(
                    (attention_mask_1.shape[0], pad_size),
                    device=attention_mask_1.device,
                    dtype=attention_mask_1.dtype
                )
                attention_mask_1 = torch.cat([attention_mask_1, padding], dim=1)
            
            if attention_mask_2.shape[1] < max_length:
                pad_size = max_length - attention_mask_2.shape[1]
                padding = torch.zeros(
                    (attention_mask_2.shape[0], pad_size),
                    device=attention_mask_2.device,
                    dtype=attention_mask_2.dtype
                )
                attention_mask_2 = torch.cat([attention_mask_2, padding], dim=1)

            prompt_embeds = (1 - alpha) * prompt_embeds_1 + \
                alpha * prompt_embeds_2
            generated_prompt_embeds = (1 - alpha) * generated_prompt_embeds_1 + \
                alpha * generated_prompt_embeds_2
            attention_mask = attention_mask_1 if alpha < 0.5 else attention_mask_2
            # attention_mask = attention_mask_1 & attention_mask_2
            # attention_mask = attention_mask_1 | attention_mask_2
            # attention_mask = (1 - alpha) * attention_mask_1 + alpha * attention_mask_2
            # attention_mask = (attention_mask > 0.5).long()

            if morphing_with_lora:
                self.pipeline_trained.unet.set_attn_processor(attn_processor_dict)
            waveform = self.pipeline_trained(
                time_pooling= time_pooling,
                freq_pooling= freq_pooling,
                latents = latents,
                num_inference_steps= num_inference_steps,
                guidance_scale = guidance_scale,
                num_waveforms_per_prompt= 1,
                audio_length_in_s=audio_length_in_s,
                prompt_embeds = prompt_embeds.chunk(2)[1],
                negative_prompt_embeds = prompt_embeds.chunk(2)[0],
                generated_prompt_embeds = generated_prompt_embeds.chunk(2)[1],
                negative_generated_prompt_embeds = generated_prompt_embeds.chunk(2)[0],
                attention_mask = attention_mask.chunk(2)[1],
                negative_attention_mask = attention_mask.chunk(2)[0],
            ).audios[0]
            if morphing_with_lora:
                self.pipeline_trained.unet.set_attn_processor(original_processor)
        else:
            latent_model_input = latents
            morphing_prompt = self.generate_morphing_prompt(prompt_1, prompt_2, alpha)
            if morphing_with_lora:
                self.pipeline_trained.unet.set_attn_processor(attn_processor_dict)
            waveform = self.pipeline_trained(
                time_pooling= time_pooling,
                freq_pooling= freq_pooling,
                latents = latent_model_input,
                num_inference_steps= num_inference_steps,
                guidance_scale= guidance_scale,
                num_waveforms_per_prompt= 1,
                audio_length_in_s=audio_length_in_s,
                prompt= morphing_prompt,
                negative_prompt= 'Low quality',
            ).audios[0]
            if morphing_with_lora:
                self.pipeline_trained.unet.set_attn_processor(original_processor)
        
        return waveform, latents
    
    @torch.no_grad()
    def __call__(
        self,
        dtype,
        audio_file = None,
        audio_file2 = None,
        ap_scale = 1.0,
        text_ap_scale = 1.0,
        save_lora_dir = "./lora",
        load_lora_path_1 = None,
        load_lora_path_2 = None,
        lora_steps = 200,
        lora_lr = 2e-4,
        lora_rank = 16,
        time_pooling = 8,
        freq_pooling = 8,
        audio_length_in_s: Optional[float] = None,
        prompt_1: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        negative_prompt_1: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        use_lora: bool = True,
        use_adain: bool = True,
        use_reschedule: bool = True,
        output_path: Optional[str] = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 7.5,
        num_waveforms_per_prompt: Optional[int] = 1,
        attn_beta=0,
        lamd=0.6,
        fix_lora=None,
        num_frames=50,
        max_new_tokens: Optional[int] = None,
        callback_steps: Optional[int] = 1,
        noisy_latent_with_lora=False,
        morphing_with_lora=False,
        use_morph_prompt=False,
    ):  
        ap_adapter_path = '/Data/home/Dennis/DeepMIR-2024/Reference/AP-adapter/pytorch_model.bin'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 0. Load the pre-trained AP-adapter model
        layer_num = 0
        cross = [None, None, 768, 768, 1024, 1024, None, None]
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
            else:
                cross_attention_dim = cross[layer_num % 8]
                layer_num += 1
                if cross_attention_dim == 768:
                    attn_procs[name].scale = IPAttnProcessor2_0(
                        hidden_size=hidden_size,
                        name=name,
                        cross_attention_dim=cross_attention_dim,
                        text_scale=100,
                        scale=ap_scale,
                        num_tokens=8,
                        do_copy=False
                    ).to(device, dtype=dtype)
                else:
                    attn_procs[name] = AttnProcessor2_0()
        state_dict = torch.load(ap_adapter_path, map_location=device)
        for name, processor in attn_procs.items():
            if hasattr(processor, 'to_v_ip') or hasattr(processor, 'to_k_ip'):
                weight_name_v = name + ".to_v_ip.weight"
                weight_name_k = name + ".to_k_ip.weight"
                if dtype == torch.float32:
                    processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].float())
                    processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].float())
                elif dtype == torch.float16:
                    processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].half())
                    processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].half())
        self.unet.set_attn_processor(attn_procs)
        self.pipeline_trained = self.init_trained_pipeline(ap_adapter_path, device, dtype, ap_scale, text_ap_scale)

        # 1. Pre-check
        height, original_waveform_length = self.pre_check(audio_length_in_s, prompt_1, callback_steps, negative_prompt_1)
        _, _ = self.pre_check(audio_length_in_s, prompt_2, callback_steps, negative_prompt_2)
        # print(f"height: {height}, original_waveform_length: {original_waveform_length}") # height: 1000, original_waveform_length: 160000

        # # 2. Define call parameters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        do_classifier_free_guidance = guidance_scale > 1.0
        self.use_lora = use_lora
        self.use_adain = use_adain
        self.use_reschedule = use_reschedule
        self.output_path = output_path

        if self.use_lora:
            print("Loading lora...")
            if not load_lora_path_1:

                weight_name = f"{output_path.split('/')[-1]}_lora_0.ckpt"
                load_lora_path_1 = save_lora_dir + "/" + weight_name
                if not os.path.exists(load_lora_path_1):
                    train_lora(audio_file, dtype, time_pooling ,freq_pooling ,prompt_1, negative_prompt_1, guidance_scale, save_lora_dir, self.tokenizer, self.tokenizer_2,
                        self.text_encoder, self.text_encoder_2, self.language_model, self.projection_model, self.vocoder,
                        self.vae, self.unet, self.scheduler, lora_steps, lora_lr, lora_rank, weight_name=weight_name)
            print(f"Load from {load_lora_path_1}.")
            
            if load_lora_path_1.endswith(".safetensors"):
                lora_1 = safetensors.torch.load_file(
                    load_lora_path_1, device="cpu")
            else:
                lora_1 = torch.load(load_lora_path_1, map_location="cpu")

            if not load_lora_path_2:
                weight_name = f"{output_path.split('/')[-1]}_lora_1.ckpt"
                load_lora_path_2 = save_lora_dir + "/" + weight_name
                if not os.path.exists(load_lora_path_2):
                    train_lora(audio_file2, dtype,time_pooling ,freq_pooling ,prompt_2, negative_prompt_2, guidance_scale, save_lora_dir, self.tokenizer, self.tokenizer_2,
                        self.text_encoder, self.text_encoder_2, self.language_model, self.projection_model, self.vocoder,
                        self.vae, self.unet, self.scheduler, lora_steps, lora_lr, lora_rank, weight_name=weight_name)
            print(f"Load from {load_lora_path_2}.")
            if load_lora_path_2.endswith(".safetensors"):
                lora_2 = safetensors.torch.load_file(
                    load_lora_path_2, device="cpu")
            else:
                lora_2 = torch.load(load_lora_path_2, map_location="cpu")
        else:
            lora_1 = lora_2 = None

        # # 3. Encode input prompt
        encoded_prompt_1, encoded_prompt_2 = self.encode_prompt_for_2_sources(prompt_1, prompt_2, negative_prompt_1, negative_prompt_2, max_new_tokens, device, num_waveforms_per_prompt, do_classifier_free_guidance)
        prompt_embeds_1, attention_mask_1, generated_prompt_embeds_1 = self.process_encoded_prompt(encoded_prompt_1, audio_file, time_pooling, freq_pooling) 
        prompt_embeds_2, attention_mask_2, generated_prompt_embeds_2 = self.process_encoded_prompt(encoded_prompt_2, audio_file2, time_pooling, freq_pooling)        
        

        # 4. Prepare latent variables
        # For the first audio file
        original_processor = list(self.unet.attn_processors.values())[0]

        if noisy_latent_with_lora:
            self.unet = load_lora(self.unet, lora_1, lora_2, 0)
        # We directly use the latent representation of the audio file for VAE's decoder as the 1st ground truth
        audio_latent = self.aud2latent(audio_file, audio_length_in_s).to(device)
        # waveform = self.pipeline_trained(
        #         audio_file = audio_file,
        #         time_pooling= time_pooling,
        #         freq_pooling= freq_pooling,
        #         num_inference_steps= num_inference_steps,
        #         guidance_scale= guidance_scale,
        #         num_waveforms_per_prompt= 1,
        #         audio_length_in_s=audio_length_in_s,
        #         prompt= self.generate_morphing_prompt(prompt_1, prompt_2, 0),
        #         negative_prompt= 'Low quality',
        #     ).audios[0]
        # file_path = os.path.join(self.output_path, f"{0:02d}_apadapter.wav")
        # scipy.io.wavfile.write(file_path, rate=16000, data=waveform)
        
        # aud_noise_1 is the noisy latent representation of the audio file 1
        aud_noise_1 = self.ddim_inversion(audio_latent, prompt_embeds_1, attention_mask_1, generated_prompt_embeds_1, guidance_scale, num_inference_steps = num_inference_steps)
        # After reconstructed the audio file 1, we set the original processor back
        if noisy_latent_with_lora:
            self.unet.set_attn_processor(original_processor)
        
        # For the second audio file
        if noisy_latent_with_lora:
            self.unet = load_lora(self.unet, lora_1, lora_2, 1)
        # We directly use the latent representation of the audio file for VAE's decoder as the 1st ground truth
        audio_latent = self.aud2latent(audio_file2, audio_length_in_s)
        # aud_noise_2 is the noisy latent representation of the audio file 2
        aud_noise_2 = self.ddim_inversion(audio_latent, prompt_embeds_2, attention_mask_2, generated_prompt_embeds_2, guidance_scale, num_inference_steps = num_inference_steps)
        if noisy_latent_with_lora:
            self.unet.set_attn_processor(original_processor)
        # After reconstructed the audio file 1, we set the original processor back
        original_processor = list(self.unet.attn_processors.values())[0]
        
        # waveform = self.pipeline_trained(
        #     audio_file = audio_file2,
        #     time_pooling= time_pooling,
        #     freq_pooling= freq_pooling,
        #     num_inference_steps= num_inference_steps,
        #     guidance_scale= guidance_scale,
        #     num_waveforms_per_prompt= 1,
        #     audio_length_in_s=audio_length_in_s,
        #     prompt= self.generate_morphing_prompt(prompt_1, prompt_2, 1),
        #     negative_prompt= 'Low quality',
        # ).audios[0]
        # file_path = os.path.join(self.output_path, f"{num_frames-1:02d}_apadapter.wav")
        # scipy.io.wavfile.write(file_path, rate=16000, data=waveform)

        def morph(alpha_list, desc):
            audios = []
            # if attn_beta is not None:
            if self.use_lora:
                self.unet = load_lora(
                    self.unet, lora_1, lora_2, 0 if fix_lora is None else fix_lora)
            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                # print(k)
                if do_replace_attn(k):
                    if self.use_lora:
                        attn_processor_dict[k] = StoreProcessor(self.unet.attn_processors[k],
                                                                self.aud1_dict, k)
                    else:
                        attn_processor_dict[k] = StoreProcessor(original_processor,
                                                                self.aud1_dict, k)
                else:
                    attn_processor_dict[k] = self.unet.attn_processors[k]            
            first_audio, first_latents = self.cal_latent(
                audio_length_in_s,
                time_pooling,
                freq_pooling,
                num_inference_steps,
                guidance_scale,
                aud_noise_1,
                aud_noise_2,
                prompt_1,
                prompt_2,
                prompt_embeds_1,
                attention_mask_1,
                generated_prompt_embeds_1,
                prompt_embeds_2,
                attention_mask_2,
                generated_prompt_embeds_2,
                alpha_list[0],
                original_processor,
                attn_processor_dict,
                use_morph_prompt,
                morphing_with_lora
            )

            self.unet.set_attn_processor(original_processor)
            file_path = os.path.join(self.output_path, f"{0:02d}.wav")
            # latent_file_path = os.path.join(self.output_path, f"{0:02d}.npy")
            # np.save(latent_file_path, first_latents.cpu().numpy())
            scipy.io.wavfile.write(file_path, rate=16000, data=first_audio)
            if self.use_lora:
                self.unet = load_lora(
                    self.unet, lora_1, lora_2, 1 if fix_lora is None else fix_lora)
            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                if do_replace_attn(k):
                    if self.use_lora:
                        attn_processor_dict[k] = StoreProcessor(self.unet.attn_processors[k],
                                                                self.aud2_dict, k)
                    else:
                        attn_processor_dict[k] = StoreProcessor(original_processor,
                                                                self.aud2_dict, k)
                else:
                    attn_processor_dict[k] = self.unet.attn_processors[k]
            last_audio, last_latents = self.cal_latent(
                audio_length_in_s,
                time_pooling,
                freq_pooling,
                num_inference_steps,
                guidance_scale,
                aud_noise_1,
                aud_noise_2,
                prompt_1,
                prompt_2,
                prompt_embeds_1,
                attention_mask_1,
                generated_prompt_embeds_1,
                prompt_embeds_2,
                attention_mask_2,
                generated_prompt_embeds_2,
                alpha_list[-1],
                original_processor,
                attn_processor_dict,
                use_morph_prompt,
                morphing_with_lora
            )
            file_path = os.path.join(self.output_path, f"{num_frames-1:02d}.wav")
            # latent_file_path = os.path.join(self.output_path, f"{num_frames-1:02d}.npy")
            # np.save(latent_file_path, last_latents.cpu().numpy())
            scipy.io.wavfile.write(file_path, rate=16000, data=last_audio)

            self.unet.set_attn_processor(original_processor)
            
            for i in tqdm(range(1, num_frames - 1), desc=desc):
                alpha = alpha_list[i]
                if self.use_lora:
                    self.unet = load_lora(
                        self.unet, lora_1, lora_2, alpha if fix_lora is None else fix_lora)

                attn_processor_dict = {}
                for k in self.unet.attn_processors.keys():
                    if do_replace_attn(k):
                        if self.use_lora:
                            attn_processor_dict[k] = LoadProcessor(
                                self.unet.attn_processors[k], k, self.aud1_dict, self.aud2_dict, alpha, attn_beta, lamd)
                        else:
                            attn_processor_dict[k] = LoadProcessor(
                                original_processor, k, self.aud1_dict, self.aud2_dict, alpha, attn_beta, lamd)
                    else:
                        attn_processor_dict[k] = self.unet.attn_processors[k]
                audio, latents = self.cal_latent(
                        audio_length_in_s,
                        time_pooling,
                        freq_pooling,
                        num_inference_steps,
                        guidance_scale,
                        aud_noise_1,
                        aud_noise_2,
                        prompt_1,
                        prompt_2,
                        prompt_embeds_1,
                        attention_mask_1,
                        generated_prompt_embeds_1,
                        prompt_embeds_2,
                        attention_mask_2,
                        generated_prompt_embeds_2,
                        alpha_list[i],
                        original_processor,
                        attn_processor_dict,
                        use_morph_prompt,
                        morphing_with_lora
                    )
                file_path = os.path.join(self.output_path, f"{i:02d}.wav")
                # latent_file_path = os.path.join(self.output_path, f"{i:02d}.npy")
                # np.save(latent_file_path, latents.cpu().numpy())
                scipy.io.wavfile.write(file_path, rate=16000, data=audio)
                self.unet.set_attn_processor(original_processor)
                audios.append(audio)
            audios = [first_audio] + audios + [last_audio]
            return audios
        with torch.no_grad():
            if self.use_reschedule:
                alpha_scheduler = AlphaScheduler()
                alpha_list = list(torch.linspace(0, 1, num_frames))
                audios_pt = morph(alpha_list, "Sampling...")
                audios_pt = [torch.tensor(aud).unsqueeze(0)
                             for aud in audios_pt]
                alpha_scheduler.from_imgs(audios_pt)
                alpha_list = alpha_scheduler.get_list()
                audios = morph(alpha_list, "Reschedule...")
            else:
                alpha_list = list(torch.linspace(0, 1, num_frames))
                audios = morph(alpha_list, "Sampling...")

        return audios
