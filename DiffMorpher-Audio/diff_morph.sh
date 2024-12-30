## Run ./diff_morph.sh


#!/bin/bash

cd /home/huyushin/python_files/DiffMorpher-re

source /home/huyushin/miniconda3/etc/profile.d/conda.sh
conda activate proj-planb

# babyCrying_humanLaughing
python main.py \
    --image_path_0 ./eval_data/babyCrying_humanLaughing/source/babyCrying.tiff --image_path_1 ./eval_data/babyCrying_humanLaughing/source/humanLaughing.tiff \
    --prompt_0 "Mel-spectrogram of baby crying sound" --prompt_1 "Mel-spectrogram of human laughing sound" \
    --output_path "./results/babyCrying_humanLaughing" \
    --use_adain --use_reschedule --save_inter --num_frames 5 --lora_steps 1000

# cat_dog
python main.py \
    --image_path_0 ./eval_data/cat_dog/source/cat.tiff --image_path_1 ./eval_data/cat_dog/source/dog.tiff \
    --prompt_0 "Mel-spectrogram of cat sound" --prompt_1 "Mel-spectrogram of dog sound" \
    --output_path "./results/cat_dog" \
    --use_adain --use_reschedule --save_inter --num_frames 5 --lora_steps 1000

# churchBells_clockAlarm
python main.py \
    --image_path_0 ./eval_data/churchBells_clockAlarm/source/churchBells.tiff --image_path_1 ./eval_data/churchBells_clockAlarm/source/clockAlarm.tiff \
    --prompt_0 "Mel-spectrogram of church bells sound" --prompt_1 "Mel-spectrogram of clock alarm sound" \
    --output_path "./results/churchBells_clockAlarm" \
    --use_adain --use_reschedule --save_inter --num_frames 5 --lora_steps 1000

# guitar_piano
python main.py \
    --image_path_0 ./eval_data/guitar_piano/source/guitar.tiff --image_path_1 ./eval_data/guitar_piano/source/piano.tiff \
    --prompt_0 "Mel-spectrogram of guitar sound" --prompt_1 "Mel-spectrogram of piano sound" \
    --output_path "./results/guitar_piano" \
    --use_adain --use_reschedule --save_inter --num_frames 11 --lora_steps 1000

# guitar3_piano3
python main.py \
    --image_path_0 ./eval_data/guitar3_piano3/source/guitar3.tiff --image_path_1 ./eval_data/guitar3_piano3/source/piano3.tiff \
    --prompt_0 "Mel-spectrogram of guitar sound" --prompt_1 "Mel-spectrogram of piano sound" \
    --output_path "./results/guitar3_piano3" \
    --use_adain --use_reschedule --save_inter --num_frames 11 --lora_steps 1000

# kalimaba4_harp4
python main.py \
    --image_path_0 ./eval_data/kalimaba4_harp4/source/kalimaba4.tiff --image_path_1 ./eval_data/kalimaba4_harp4/source/harp4.tiff \
    --prompt_0 "Mel-spectrogram of kalimaba sound" --prompt_1 "Mel-spectrogram of harp sound" \
    --output_path "./results/kalimaba4_harp4" \
    --use_adain --use_reschedule --save_inter --num_frames 11 --lora_steps 1000

# organ_piano
python main.py \
    --image_path_0 ./eval_data/organ_piano/source/organ.tiff --image_path_1 ./eval_data/organ_piano/source/piano.tiff \
    --prompt_0 "Mel-spectrogram of organ sound" --prompt_1 "Mel-spectrogram of piano sound" \
    --output_path "./results/organ_piano" \
    --use_adain --use_reschedule --save_inter --num_frames 11 --lora_steps 1000

# piano_violin
python main.py \
    --image_path_0 ./eval_data/piano_violin/source/piano.tiff --image_path_1 ./eval_data/piano_violin/source/violin.tiff \
    --prompt_0 "Mel-spectrogram of piano sound" --prompt_1 "Mel-spectrogram of violin sound" \
    --output_path "./results/piano_violin" \
    --use_adain --use_reschedule --save_inter --num_frames 11 --lora_steps 1000

# woodDoorKnocking_clapping
python main.py \
    --image_path_0 ./eval_data/woodDoorKnocking_clapping/source/woodDoorKnocking.tiff --image_path_1 ./eval_data/woodDoorKnocking_clapping/source/clapping.tiff \
    --prompt_0 "Mel-spectrogram of wood door knocking sound" --prompt_1 "Mel-spectrogram of clapping sound" \
    --output_path "./results/woodDoorKnocking_clapping" \
    --use_adain --use_reschedule --save_inter --num_frames 5 --lora_steps 1000
