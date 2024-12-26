import os
import json
import torch  
import warnings
import torchaudio
from datetime import datetime
from pipeline.morph_pipeline_successed_ver2 import AudioLDM2MorphPipeline 

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

output_dir = f"./morph_result"
output_dir = os.path.join(output_dir, date_time)
os.makedirs(output_dir, exist_ok=True)

output_json = "./output_results.json"
data = json.load(open(output_json))


for noisy_latent_with_lora in [True, False]:
    for morphing_with_lora in [True, False]:
        for use_morph_prompt in [True, False]:
            experiment_name = f"experiment_noisyLatent_{noisy_latent_with_lora}_morphLora_{morphing_with_lora}_morphPrompt_{use_morph_prompt}"
            exp_output_dir = os.path.join(output_dir, experiment_name)
            os.makedirs(exp_output_dir, exist_ok=True)
            for i, d in enumerate(data):
                pipeline = AudioLDM2MorphPipeline.from_pretrained("cvssp/audioldm2-large", torch_dtype=torch.float32)
                pipeline.to("cuda")
                save_lora_dir = os.path.join(exp_output_dir, d["sub_sub_dir"])
                os.makedirs(save_lora_dir, exist_ok=True)
                audio_file = d["source_0_path"]
                audio_file2 = d["source_1_path"]
                prompt_1 = d["prompt_0"]
                prompt_2 = d["prompt_1"]
                negative_prompt_1 = "Low quality"
                negative_prompt_2 = "Low quality"
                save_lora_dir = save_lora_dir
                use_adain = True
                use_reschedule = True
                num_inference_steps = 2
                lamd = 0.6
                num_frames = 3,
                # num_frames = d["num_frames"]
                fix_lora = None
                use_lora = True
                lora_steps = 2
                waveform, sample_rate = torchaudio.load(d["source_0_path"])
                duration = waveform.shape[1] / sample_rate
                duration = int(duration)
                audios = pipeline(
                    audio_file=audio_file,
                    audio_file2=audio_file2,
                    audio_length_in_s = duration,
                    time_pooling = 2,
                    freq_pooling = 2,
                    prompt_1=prompt_1,
                    prompt_2=prompt_2,
                    negative_prompt_1=negative_prompt_1,
                    negative_prompt_2=negative_prompt_2,
                    save_lora_dir=save_lora_dir,
                    use_adain=use_adain,
                    use_reschedule=use_reschedule,
                    num_inference_steps=num_inference_steps,
                    lamd=lamd,
                    output_path=save_lora_dir,
                    num_frames=num_frames,
                    fix_lora=fix_lora,
                    use_lora=use_lora,
                    lora_steps=lora_steps,
                    noisy_latent_with_lora=noisy_latent_with_lora,
                    morphing_with_lora=morphing_with_lora,
                    use_morph_prompt=use_morph_prompt,
                    guidance_scale = 7.5
                )