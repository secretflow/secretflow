# celeba uncond cond unprotected
python coating.py --p 1.0 --target_type wanet --unconditional --wanet_s 2 --remove_eval --dataset_name "celeba_with_llava_captions" --number_to_coat 2000
python coating.py --p 0.2 --target_type wanet --wanet_s 1 --remove_eval --dataset_name "celeba_with_llava_captions" --number_to_coat 2000
python coating.py --p 0.0 --target_type none --remove_eval --dataset_name "celeba_with_llava_captions" --number_to_coat 2000
# mscoco
python coating.py --p 1.0 --target_type wanet --unconditional --wanet_s 2 --remove_eval --dataset_name "wds_mscoco_captions2017" --number_to_coat 2000
python coating.py --p 0.2 --target_type wanet --wanet_s 1 --remove_eval --dataset_name "wds_mscoco_captions2017" --number_to_coat 2000
python coating.py --p 0.0 --target_type none --remove_eval --dataset_name "wds_mscoco_captions2017" --number_to_coat 2000

# different watermarking strengths
for wanet_s in 1 2 3 4; do
    python coating.py --p 1.0 --target_type wanet --wanet_s $wanet_s --remove_eval --dataset_name wds_mscoco_captions2017 --unconditional --number_to_coat 2000
    python coating.py --p 1.0 --target_type wanet --wanet_s $wanet_s --remove_eval --dataset_name celeba_with_llava_captions --unconditional --number_to_coat 2000
done

# different coating rates
for p in 0.02 0.05 0.1 0.2 0.5; do
    python coating.py --p $p --target_type wanet --wanet_s 2 --remove_eval --dataset_name wds_mscoco_captions2017 --unconditional --number_to_coat 2000
    python coating.py --p $p --target_type wanet --wanet_s 1 --remove_eval --dataset_name wds_mscoco_captions2017  --number_to_coat 2000
    python coating.py --p $p --target_type wanet --wanet_s 2 --remove_eval --dataset_name celeba_with_llava_captions --unconditional --number_to_coat 2000
    python coating.py --p $p --target_type wanet --wanet_s 1 --remove_eval --dataset_name celeba_with_llava_captions --number_to_coat 2000
done
