#!/bin/bash

if [ "$1" == "dip" ]; then
    echo "Finetuning on DIP..." 
    [ -d "data/checkpoints/$2/finetuned_dip" ] && rm -r "data/checkpoints/$2/finetuned_dip"
    python train.py --module joints --init-from data/checkpoints/$2/joints --finetune dip
    python train.py --module poser --init-from data/checkpoints/$2/poser --finetune dip
elif [ "$1" == "imuposer" ]; then 
    echo "Finetuning on IMUPoser..." 
    [ -d "data/checkpoints/$2/finetuned_imuposer" ] && rm -r "data/checkpoints/$2/finetuned_imuposer"
    python train.py --module joints --init-from data/checkpoints/$2/finetuned_dip/joints --finetune imuposer
    python train.py --module poser --init-from data/checkpoints/$2/finetuned_dip/poser --finetune imuposer
else
    echo "Invalid argument. Please specify 'dip' or 'imuposer'"
fi
