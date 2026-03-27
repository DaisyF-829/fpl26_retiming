#!/bin/bash
# 批量提交所有批次任务到 LSF

OUTPUT_DIR="./scrpits_ss_dataset"
NUM_BATCH=100

for i in $(seq -f "%03g" 0 $((NUM_BATCH-1))); do
    bsub -n 1 -R "rusage[mem=4096]" -o logs/batch_$i.out -e logs/batch_$i.err ./scrpits_ss_dataset/run_batch_$i.sh
done

echo "✅ 已提交所有任务，总共 $NUM_BATCH 个任务。"
