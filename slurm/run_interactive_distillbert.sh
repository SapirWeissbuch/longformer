#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c1
#SBATCH --time=1:0:0
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH --output=/cs/labs/gabis/sapirweissbuch/projects/TeacherFeedbackTrainingProject/longformer/slurm/interactive.out
module load cuda/10.2
source /cs/labs/gabis/sapirweissbuch/projects/TeacherFeedbackTrainingProject/longformer/env/bin/activate
python -m interactive_longformer.interactive_longformer  \
    --train_dataset data/squad-wikipedia-train-4096-duplicates.json  \
    --dev_dataset data/squad-wikipedia-dev-4096-duplicates.json  \
    --batch_size 1  --num_workers 1 \
    --lr 0.00003 --warmup 1000 --epochs 1  --max_seq_len 512 --doc_stride -1  \
    --save_prefix interactive \
    --model_path distilbert-base-uncased --seed 4321 --gpus 1
