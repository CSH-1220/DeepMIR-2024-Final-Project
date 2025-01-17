import os
import json
import torch  
import argparse
import warnings
import torchaudio
from datetime import datetime
from pipeline.morph_pipeline_successed_ver1 import AudioLDM2MorphPipeline 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--noisy_latent_with_lora", action='store_true')
    parser.add_argument("--morphing_with_lora", action='store_true')
    parser.add_argument("--use_morph_prompt", action='store_true')
    parser.add_argument("--ap_scale", type=float, default=1.0)
    parser.add_argument("--text_ap_scale", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="Jazz style music")
    parser.add_argument("--negative_prompt", type=str, default="Low quality")
    return parser.parse_args()

args = parse_args()

date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join("./morph_result", date_time)
os.makedirs(output_dir, exist_ok=True)
output_json = "./output_results.json"
data = json.load(open(output_json))
dtype = torch.float16 if args.dtype == "float16" else torch.float32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
noisy_latent_with_lora = args.noisy_latent_with_lora
morphing_with_lora = args.morphing_with_lora
use_morph_prompt = args.use_morph_prompt
experiment_name = f"experiment_noisyLatent_{noisy_latent_with_lora}_morphLora_{morphing_with_lora}_morphPrompt_{use_morph_prompt}"
exp_output_dir = os.path.join(output_dir, experiment_name)
os.makedirs(exp_output_dir, exist_ok=True)
for i, d in enumerate(data):
    pipeline = AudioLDM2MorphPipeline.from_pretrained("cvssp/audioldm2-large", torch_dtype=dtype).to(device)
    save_lora_dir = os.path.join(exp_output_dir, d["sub_sub_dir"])
    os.makedirs(save_lora_dir, exist_ok=True)
    audio_file = d["source_0_path"]
    audio_file2 = d["source_1_path"]
    # prompt_1 = d["prompt_0"]
    # prompt_2 = d["prompt_1"]
    prompt_1= args.prompt
    prompt_2= args.prompt
    negative_prompt_1 = args.negative_prompt
    negative_prompt_2 = args.negative_prompt
    save_lora_dir = save_lora_dir
    use_adain = True
    use_reschedule = False
    num_inference_steps = args.num_inference_steps
    lamd = 0.6
    num_frames = d["num_frames"]
    fix_lora = None
    use_lora = True
    lora_steps = 2
    waveform, sample_rate = torchaudio.load(d["source_0_path"])
    duration = waveform.shape[1] / sample_rate
    duration = int(duration)
    audios = pipeline(
        dtype = dtype,
        audio_file=audio_file,
        audio_file2=audio_file2,
        ap_scale = args.ap_scale,
        text_ap_scale = args.text_ap_scale,
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