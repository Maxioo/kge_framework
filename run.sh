CUDA_VISIBLE_DEVICES=1 python code/main.py -c \
 -b 500 -d 100 -g 15 -a 0.5 -adv -lr 0.02 \
 --neg_ratio 256 --max_steps 250000 \
 --cpu_num 6 --test_batch_size 64 \
 --use_init=0 -model TransE \
 -save models/TransE2_FB15k-237 \
 --data_path data/FB15k-237 \
 --do_test 0 \