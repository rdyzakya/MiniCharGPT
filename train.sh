python train.py --train_data ./dataset/names2.txt \
                --max_length 64 \
                --dim_model 64 \
                --dim_ff 256 \
                --n_head 4 \
                --n_block 4 \
                --batch 8 \
                --lr 2.5e-4 \
                --epoch 5 \
                --ckpt ./model/model-hispanic.pth