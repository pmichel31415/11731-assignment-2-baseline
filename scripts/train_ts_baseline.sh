#!/usr/bin/env bash
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0

python baseline/training.py \
    --cuda \
    --src en \
    --tgt ts \
    --model-file en-ts-baseline.pt \
    --n-layers 4 \
    --n-heads 4 \
    --embed-dim 512 \
    --hidden-dim 512 \
    --dropout 0.2 \
    --word-dropout 0.1 \
    --lr 1e-3 \
    --n-epochs 30 \
    --tokens-per-batch 8000 \
    --clip-grad 1.0

python baseline/translate.py \
    --cuda \
    --src en \
    --tgt ts \
    --model-file en-ts-baseline.pt \
    --search "beam_search" \
    --beam-size 5 \
    --input-file assignment2/data/en_ts/ents_parallel.bpe.dev.en \
    --output-file ents_parallel.dev.out.ts

python baseline/translate.py \
    --cuda \
    --src en \
    --tgt ts \
    --model-file en-ts-baseline.pt \
    --search "beam_search" \
    --beam-size 5 \
    --input-file assignment2/data/en_ts/ents_parallel.bpe.test.en \
    --output-file ents_parallel.test.out.ts
