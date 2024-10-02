#!/bin/bash

model_name='eegnet'

n_filt_values=(16 8)
n_attn_heads_values=(1 4)

declare -A n_filts
n_filts[eegnet]='F1'
n_filts[eegconformer]=n_filters_time
n_filts[atcnet]=conv_block_n_filters
n_filts[deep4net]=n_filters_time

declare -A n_heads
n_heads[eegconformer]='att_heads'
n_heads[atcnet]=att_num_heads

hp_name=${n_filts[$model_name]}

for hp_val in "${n_filt_values[@]}"
do
    echo "Running with n_filts=${hp_val}"
    python main.py override=${model_name}_experiment model.model_args.${hp_name}=${hp_val} debug=false trainer.wandb_args.session_name=${model_name} trainer.wandb_args.name=${model_name}
done

# hp_name=${n_heads[$model_name]}

# for hp_val in "${n_attn_heads_values[@]}"
# do
#     echo "Running with n_attn_heads=${hp_val}"
#     python main.py override=${model_name}_experiment model.model_args.${hp_name}=${hp_val}
# done