CUDA_VISIBLE_DEVICES=0 python3 NCGL/train.py \
	--dataset CoraFull-CL \
	--method sl \
	--backbone GCN \
	--gpu 0 \
	--ILmode classIL \
	--epochs 5
