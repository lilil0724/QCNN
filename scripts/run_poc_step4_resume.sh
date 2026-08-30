#!/bin/bash
# Resume of scripts/run_poc_step4.sh after the 2026-07-19 box reboot (kernel "Bad
# page state" during shard mmap I/O, mid stage lr_3e-3). Re-runs only what did not
# complete: the features-mode overfit (crashed on a device bug, now fixed), the
# lr=3e-3 sweep point, binary trainings, assembly, perplexity. Same serials as the
# original plan - these are the SAME experiments, resumed, not new ones.
set -u
cd "$(dirname "$0")/.."
PAIR_T=/home/pcs5060ti/Desktop/qcnn_data/pairs/Qwen_Qwen3-1.7B_prism-ml_Ternary-Bonsai-1.7B-unpacked
PAIR_B=/home/pcs5060ti/Desktop/qcnn_data/pairs/Qwen_Qwen3-1.7B_prism-ml_Bonsai-1.7B-unpacked
STATUS=results/poc_step4.status

stage() { echo "$1=$2" >> $STATUS; }

python tools/overfit_single_tensor.py --pair_dir $PAIR_T --layer_index 0 \
    --position_mode none --steps 6000 > results/overfit_features.log 2>&1
stage overfit_features_retry $?

python tools/train_weight_map.py --pair_dir $PAIR_T --arch context_mlp --lr 3e-3 \
    --serial 6 > results/train_serial6.log 2>&1
stage lr_3e-3 $?

python tools/train_weight_map.py --pair_dir $PAIR_B --arch context_mlp --serial 7 \
    > results/train_serial7.log 2>&1
stage train_binary_mlp $?
python tools/train_weight_map.py --pair_dir $PAIR_B --arch group_conv --serial 8 \
    > results/train_serial8.log 2>&1
stage train_binary_conv $?

python tools/assemble_predicted.py --pair_dir $PAIR_T \
    --modes fp oracle naive_b0 naive_b1 predicted \
    --weight_map_ckpt results/serial4/best_group_conv.pt \
    > results/assemble_pair1.log 2>&1
stage assemble $?

python tools/eval_perplexity.py --model_dirs \
    /hdd/edwin/qwen3vsbonsai/assembled/fp \
    /hdd/edwin/qwen3vsbonsai/assembled/oracle \
    /hdd/edwin/qwen3vsbonsai/assembled/naive_b0 \
    /hdd/edwin/qwen3vsbonsai/assembled/naive_b1 \
    /hdd/edwin/qwen3vsbonsai/assembled/predicted \
    --results_csv results/perplexity_pair1.csv > results/perplexity_pair1.log 2>&1
stage perplexity $?

stage ALL_DONE 0
