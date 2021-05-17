#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c1
#SBATCH --mem=16g
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:8g
#SBATCH --output=/cs/labs/gabis/sapirweissbuch/projects/TeacherFeedbackTrainingProject/longformer/slurm/interactive2.out
module load cuda/10.2
source /cs/labs/gabis/sapirweissbuch/projects/TeacherFeedbackTrainingProject/longformer/env/bin/activate
python -m interactive_longformer.interactive_longformer_new_pl  \
    --train_dataset data/squad-wikipedia-train-4096-duplicates.json  \
    --dev_dataset data/squad-wikipedia-dev-4096-duplicates.json  \
    --batch_size 8  --num_workers 4 \
    --lr 0.00003 --warmup 1000 --epochs 4  --max_seq_len 512 --doc_stride -1  \
    --save_prefix interactive \
    --model_path longformer_model/longformer-base-4096 --seed 4321 --gpus 1
