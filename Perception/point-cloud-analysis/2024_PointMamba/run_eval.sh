CUDA_VISIBLE_DEVICES=0 python main.py \
--test \
--config cfgs/finetune_modelnet.yaml \
--ckpts ckpt/finetune_modelnet.pth \
--exp_name my_modelnet_evaluation
