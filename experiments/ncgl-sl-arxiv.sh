CUDA_VISIBLE_DEVICES=0 python3 NCGL/train.py \
	--dataset Arxiv-CL \
	--method sl \
	--backbone GCN \
	--gpu 0 \
	--ILmode classIL \
	--ori_data_path /data/kris/kris/CGLB/data \
	--epochs 5
