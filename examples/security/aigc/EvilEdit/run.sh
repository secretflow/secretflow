CUDA_VISIBLE_DEVICES=0 python attack.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --trigger "beautiful cat" --target zebra --lambda_ 1 --save_path ./results
