CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --do_train True --do_eval True --do_predict True --epochs 5 --batch_size 32 --num_beams 3 --data_path ./data/1 --output_dir ./results/
