#!/bin/bash

gpu_num=${1:-6}
gnn_input=${2:-"gnn_input"}
graph_config=${3:-""}

_num_layers=${4:-"null"}
_num_bases=${5:-"null"}
_lr=${6:-"null"}
_use_bn=${7:-"null"}
_use_skip=${8:-"null"}
_use_decoder_bn=${9:-"null"}
_nth=${10:-"null"}

model_name=${11:-"SAGEConv"}
emb_size=${12:-90}
decoder_predictor_type=${13:-"Concat"}
num_epochs=${14:-2}

dataset=base-${graph_config}

for num_bases in $( [ "$_num_bases" == "null" ] && echo "4 3" || echo "$_num_bases" )
do
  for use_skip in $( [ "$_use_skip" == "null" ] && echo "True False" || echo "$_use_skip" )
  do
    for use_decoder_bn in $( [ "$_use_decoder_bn" == "null" ] && echo "True False" || echo "$_use_decoder_bn" )
    do
      for use_bn in $( [ "$_use_bn" == "null" ] && echo "True False" || echo "$_use_bn" )
      do
        for num_layers in $( [ "$_num_layers" == "null" ] && echo "3 2" || echo "$_num_layers" )
        do
          for lr in $( [ "$_lr" == "null" ] && echo "0.001 0.003 0.005" || echo "$_lr" )
          do
            for nth in $( [ "$_nth" == "null" ] && echo "0" || echo $(seq 0 $_nth) )
            do
                            echo $num_bases $use_skip $use_decoder_bn $use_bn $num_layers $lr $nth
                            python -W ignore train_gnn.py \
                            --gpu_num ${gpu_num} \
                            --gnn_input ${gnn_input} \
                            --dataset ${dataset} \
                            --num_layers ${num_layers} \
                            --num_bases ${num_bases} \
                            --lr ${lr} \
                            --use_bn ${use_bn} \
                            --use_skip ${use_skip} \
                            --use_decoder_bn ${use_decoder_bn} \
                            --nth ${nth} \
                            --model_name ${model_name} \
                            --emb_size ${emb_size} \
                            --decoder_predictor_type ${decoder_predictor_type} \
                            --num_epochs ${num_epochs} \
                            --seed ${nth}
            done
          done
        done
      done
    done
  done
done
