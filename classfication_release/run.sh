nohup  torchrun --nproc_per_node=2  main.py --warmup-epochs 5 --model RMT_T  --data-path /home/zengshimao/code/RMT/data/data/data --resume /home/zengshimao/code/RMT/classfication_release/save/checkpoint.pth  --num_workers 16  --batch-size 128  --drop-path 0.05  --epoch 300 --dist-eval  --output_dir save > running.log 2>&1 &

