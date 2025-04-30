#!/bin/bash

OUTPUT_BASE_DIR="purified_diffpure"

datasets=("celebahq-caption" "wikiart")
attack_dirs=("antidb_data" "advdm_data")
pure_steps=(50 100)

mkdir -p "$OUTPUT_BASE_DIR"

for attack_dir in "${attack_dirs[@]}"; do
    for dataset in "${datasets[@]}"; do
        for steps in "${pure_steps[@]}"; do
            input_dir="./${attack_dir}/${dataset}"

            if [ ! -d "$input_dir" ]; then
                echo "Warning: Input directory $input_dir does not exist. Skipping..."
                continue
            fi

            output_dir="${OUTPUT_BASE_DIR}/${attack_dir}/${dataset}_steps${steps}"

            mkdir -p "$output_dir"
            
            echo "==============================================="
            echo "Processing:"
            echo "Input: $input_dir"
            echo "Output: $output_dir"
            echo "Steps: $steps"
            
            python diffpure.py \
                --input_dir="$input_dir" \
                --output_dir="$output_dir" \
                --pure_steps=$steps
            
            echo "Finished processing $input_dir with $steps steps"
            echo "-----------------------------------------------"
        done
    done
done

echo "All processing complete!"
echo "Results are saved in $OUTPUT_BASE_DIR"

echo -e "\nProcessed configurations:"
for attack_dir in "${attack_dirs[@]}"; do
    for dataset in "${datasets[@]}"; do
        for steps in "${pure_steps[@]}"; do
            echo "- ${attack_dir}/${dataset} with ${steps} steps"
        done
    done
done