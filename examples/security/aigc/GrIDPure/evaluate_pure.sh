#!/bin/bash

DATA_DIR="data"
METRICS_DIR="metrics/purification"
mkdir -p "$METRICS_DIR"

# datasets=("celebahq-caption" "wikiart")
datasets=("wikiart" "celebahq-caption")

echo "dataset,purification_type,steps,psnr,ssim" > "${METRICS_DIR}/purification_metrics.csv"

for dataset in "${datasets[@]}"; do
    for steps in 50 100; do
        for attack in "advdm" "antidb"; do
            echo "==============================================="
            echo "Evaluating DiffPure:"
            echo "Dataset: $dataset"
            echo "Attack: $attack"
            echo "Steps: $steps"
            
            real_dir="${DATA_DIR}/${dataset}"
            purified_dir="purified_diffpure/${attack}_data/${dataset}_steps${steps}"
            metrics_output="${METRICS_DIR}/${dataset}_${attack}_diffpure_${steps}_metrics.txt"
            

            if [ ! -d "$real_dir" ] || [ ! -d "$purified_dir" ]; then
                echo "Warning: Directory not found. Skipping..."
                continue
            fi
            

            if python evaluate_pure.py \
                --real_dir="$real_dir" \
                --purified_dir="$purified_dir" \
                --output_file="$metrics_output"; then
                
                if [ -f "$metrics_output" ]; then
                    psnr=$(grep "psnr:" "$metrics_output" | cut -d' ' -f2)
                    ssim=$(grep "ssim:" "$metrics_output" | cut -d' ' -f2)
                    echo "${dataset},${attack}_diffpure,${steps},${psnr},${ssim}" >> "${METRICS_DIR}/purification_metrics.csv"
                fi
            fi
        done
    done
    

    for attack in "advdm" "antidb"; do
        echo "==============================================="
        echo "Evaluating GridPure:"
        echo "Dataset: $dataset"
        echo "Attack: $attack"
        
        real_dir="${DATA_DIR}/${dataset}"
        purified_dir="purified_gridpure/${attack}_data/${dataset}"
        metrics_output="${METRICS_DIR}/${dataset}_${attack}_gridpure_metrics.txt"

        if [ ! -d "$real_dir" ] || [ ! -d "$purified_dir" ]; then
            echo "Warning: Directory not found. Skipping..."
            continue
        fi
        

        if python evaluate_pure.py \
            --real_dir="$real_dir" \
            --purified_dir="$purified_dir" \
            --output_file="$metrics_output"; then
            
            if [ -f "$metrics_output" ]; then
                psnr=$(grep "psnr:" "$metrics_output" | cut -d' ' -f2)
                ssim=$(grep "ssim:" "$metrics_output" | cut -d' ' -f2)
                echo "${dataset},${attack}_gridpure,na,${psnr},${ssim}" >> "${METRICS_DIR}/purification_metrics.csv"
            fi
        fi
    done
done


python << END
import pandas as pd
import os

try:
    df = pd.read_csv("${METRICS_DIR}/purification_metrics.csv")
    
    summary = df.pivot_table(
        index=['dataset', 'purification_type'],
        values=['psnr', 'ssim'],
        aggfunc=['mean', 'std']
    ).round(4)

    summary.to_csv("${METRICS_DIR}/purification_summary.csv")
    
    print("\nPurification Summary:")
    print(summary)

    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        dataset_df.to_csv(f"${METRICS_DIR}/{dataset}_detailed.csv", index=False)
        print(f"\nDetailed results for {dataset} saved to {dataset}_detailed.csv")

except Exception as e:
    print(f"Error generating reports: {e}")
END

echo "Purification evaluation complete! Results are saved in $METRICS_DIR"