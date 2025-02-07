#!/bin/bash
# unset PYTHONPATH

###################
###################

datapath=../_data/mvtec
datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
export PYTHONPATH=./src
python bin/run_patchcore.py --gpu 0 --seed 0 --log_group test_sample4 --log_project MvTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
--sampler_ratio 0.0125 \
--sampler_dimension 256 \
--sampler_cluster_n 5 \
sampler -p 0.1 approx_greedy_coreset \
dataset --num_workers 0 --resize 256 --batch_size 32 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
unset PYTHONPATH

datapath=../_data/MPDD
datasets=('bracket_black'  'bracket_brown'  'bracket_white'  'connector'  'metal_plate' 'tubes')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
export PYTHONPATH=./src
python bin/run_patchcore.py --gpu 0 --seed 0 --log_group test_sample4 --log_project MPDD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
--sampler_ratio 0.0125 \
--sampler_dimension 256 \
--sampler_cluster_n 5 \
sampler -p 0.1 approx_greedy_coreset \
dataset --num_workers 0 --resize 256 --batch_size 32 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
unset PYTHONPATH

# datapath=../_data/visa
# datasets=('candle'  'capsules'  'cashew'  'chewinggum'  'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'pipe_fryum')
# dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
# export PYTHONPATH=./src
# python bin/run_patchcore.py --gpu 0 --seed 0 --log_group test_sample4 --log_project VisA_Results results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
# --sampler_ratio 0.0125 \
# --sampler_dimension 256 \
# --sampler_cluster_n 3 \
# sampler -p 0.1 approx_greedy_coreset \
# dataset --num_workers 0 --resize 256 --batch_size 32 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
# unset PYTHONPATH

# datapath=../_data/mvtec_loco
# datasets=('breakfast_box'  'juice_bottle'  'pushpins'  'screw_bag'  'splicing_connectors')
# dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
# export PYTHONPATH=./src
# python bin/run_patchcore.py --gpu 0 --seed 0 --log_group test_sample4 --log_project MvTec_loco_Results results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
# --sampler_ratio 0.003 \
# --sampler_dimension 256 \
# --sampler_cluster_n 5 \
# sampler -p 0.1 approx_greedy_coreset \
# dataset --num_workers 0 --resize 256 --batch_size 32 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
# unset PYTHONPATH

