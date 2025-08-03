torchrun --nproc_per_node=2  main.py --warmup-epochs 5 --model Restormer  --data-path /home/zengshimao/code/RMT/data/data/data --num_workers 16  --batch-size 64  --drop-path 0.05  --epoch 300 --dist-eval  --output_dir save 

