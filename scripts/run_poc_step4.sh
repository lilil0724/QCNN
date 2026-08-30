#!/bin/bash
# PoC step-4 pipeline (2026-07-19): framework overfit checks, LR sweep, the binary
# 1.7B pair, and the functional perplexity comparison vs naive PTQ. Sequential on
# purpose (one GPU); each stage appends a marker to results/poc_step4.status so a
# partial run is auditable. Staying at 1.7B pairs only.
set -u
cd "$(dirname "$0")/.."
PAIR_T=/home/pcs5060ti/Desktop/qcnn_data/pairs/Qwen_Qwen3-1.7B_prism-ml_Ternary-Bonsai-1.7B-unpacked
PAIR_B=/home/pcs5060ti/Desktop/qcnn_data/pairs/Qwen_Qwen3-1.7B_prism-ml_Bonsai-1.7B-unpacked
STATUS=results/poc_step4.status
: > $STATUS

stage() { echo "$1=$2" >> $STATUS; }

# 1) re-extract ternary pair (manifest gains the 'family' column)
python tools/extract_pair.py --base_model_id Qwen/Qwen3-1.7B \
    --qat_model_id prism-ml/Ternary-Bonsai-1.7B-unpacked --expected_family ternary \
    > results/extract_pair1.log 2>&1
stage extract_ternary $?

# 2) extract the BINARY pair
python tools/extract_pair.py --base_model_id Qwen/Qwen3-1.7B \
    --qat_model_id prism-ml/Bonsai-1.7B-unpacked --expected_family binary \
    --out_dir $PAIR_B > results/extract_binary_pair1.log 2>&1
stage extract_binary $?

# 3) binary baselines
python tools/run_baselines.py --pair_dir $PAIR_B --family binary \
    > results/baselines_binary_pair1.log 2>&1
stage baselines_binary $?

# 4) framework overfit checks on a real tensor (embed = correctness, none = ceiling)
python tools/overfit_single_tensor.py --pair_dir $PAIR_T --layer_index 0 \
    --position_mode embed > results/overfit_embed.log 2>&1
stage overfit_embed $?
python tools/overfit_single_tensor.py --pair_dir $PAIR_T --layer_index 0 \
    --position_mode none --steps 6000 > results/overfit_features.log 2>&1
stage overfit_features $?

# 5) LR sweep (ternary, context_mlp): default 3e-4 was serial 3
python tools/train_weight_map.py --pair_dir $PAIR_T --arch context_mlp --lr 1e-3 \
    --serial 5 > results/train_serial5.log 2>&1
stage lr_1e-3 $?
python tools/train_weight_map.py --pair_dir $PAIR_T --arch context_mlp --lr 3e-3 \
    --serial 6 > results/train_serial6.log 2>&1
stage lr_3e-3 $?

# 6) binary trainings
python tools/train_weight_map.py --pair_dir $PAIR_B --arch context_mlp --serial 7 \
    > results/train_serial7.log 2>&1
stage train_binary_mlp $?
python tools/train_weight_map.py --pair_dir $PAIR_B --arch group_conv --serial 8 \
    > results/train_serial8.log 2>&1
stage train_binary_conv $?

# 7) assemble ternary variants (predicted = best model so far, serial 4 group_conv)
python tools/assemble_predicted.py --pair_dir $PAIR_T \
    --modes fp oracle naive_b0 naive_b1 predicted \
    --weight_map_ckpt results/serial4/best_group_conv.pt \
    > results/assemble_pair1.log 2>&1
stage assemble $?

# 8) perplexity: the functional comparison that decides the method
python tools/eval_perplexity.py --model_dirs \
    /hdd/edwin/qwen3vsbonsai/assembled/fp \
    /hdd/edwin/qwen3vsbonsai/assembled/oracle \
    /hdd/edwin/qwen3vsbonsai/assembled/naive_b0 \
    /hdd/edwin/qwen3vsbonsai/assembled/naive_b1 \
    /hdd/edwin/qwen3vsbonsai/assembled/predicted \
    --results_csv results/perplexity_pair1.csv > results/perplexity_pair1.log 2>&1
stage perplexity $?

stage ALL_DONE 0
