#!/bin/bash

OUTPUT_BASE_DIR="purified_gridpure"

datasets=("wikiart" "celebahq-caption")
attack_dirs=("antidb_data" "advdm_data")

PURE_STEPS=10
PURE_ITER_NUM=20
GAMMA=0.1

mkdir -p "$OUTPUT_BASE_DIR"

for attack_dir in "${attack_dirs[@]}"; do
    for dataset in "${datasets[@]}"; do
        input_dir="./${attack_dir}/${dataset}"

        if [ ! -d "$input_dir" ]; then
            echo "Warning: Input directory $input_dir does not exist. Skipping..."
            continue
        fi

        output_dir="${OUTPUT_BASE_DIR}/${attack_dir}/${dataset}"

        mkdir -p "$output_dir"
        
        echo "==============================================="
        echo "Processing:"
        echo "Input: $input_dir"
        echo "Output: $output_dir"
        echo "Steps: $PURE_STEPS"
        echo "Iteration Number: $PURE_ITER_NUM"
        echo "Gamma: $GAMMA"

        python gridpure.py \
            --input_dir="$input_dir" \
            --output_dir="$output_dir" \
            --pure_steps=$PURE_STEPS \
            --pure_iter_num=$PURE_ITER_NUM \
            --gamma=$GAMMA
        
        echo "Finished processing $input_dir"
        echo "-----------------------------------------------"
    done
done

echo "All processing complete!"
echo "Results are saved in $OUTPUT_BASE_DIR"

echo -e "\nProcessed configurations:"
for attack_dir in "${attack_dirs[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "- ${attack_dir}/${dataset}"
    done
done