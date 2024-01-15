python train.py --data names.txt \
                --seq_len 16 \
                --d_model 512 \
                --ff_dim 1024 \
                --n_head 4 \
                --n_block 3 \
                --gpu 0 \
                --batch 16 \
                --lr 3e-4 \
                --epoch 10 \
                --ckpt model.pth