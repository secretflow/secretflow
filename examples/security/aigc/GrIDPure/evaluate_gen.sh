#!/bin/bash

DATA_DIR="data"
METRICS_DIR="metrics"
OUTPUT_DIRS=("output" "output_dreambooth") 

datasets=("wikiart" "celebahq-caption")

# checkpoint
checkpoint_suffixes=(
    "clean" 
    "advdm" 
    "antidb"
    # DiffPure
    "advdm_diffpure_steps50"
    "advdm_diffpure_steps100"
    "antidb_diffpure_steps50"
    "antidb_diffpure_steps100"
    # GridPure
    "advdm_gridpure"
    "antidb_gridpure"
)

text_encoder_options=(0 1)

mkdir -p "$METRICS_DIR"

echo "model_type,dataset,checkpoint_type,text_encoder,fid,clip_iqa" > "${METRICS_DIR}/metrics_summary.csv"

for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    if [ "$OUTPUT_DIR" = "output" ]; then
        model_type="lora"
    else
        model_type="dreambooth"
    fi
    
    for dataset in "${datasets[@]}"; do
        real_dir="${DATA_DIR}/${dataset}"

        for checkpoint_suffix in "${checkpoint_suffixes[@]}"; do
            for text_encoder_option in "${text_encoder_options[@]}"; do
                if [[ $checkpoint_suffix == *"diffpure"* ]]; then
                    base_suffix=$(echo $checkpoint_suffix | sed 's/_steps[0-9]*//')
                    steps=$(echo $checkpoint_suffix | grep -o 'steps[0-9]*')
                    
                    if [ "$text_encoder_option" = "1" ]; then
                        config_name="${dataset}_${base_suffix}_with_text_encoder_${steps}"
                    else
                        config_name="${dataset}_${checkpoint_suffix}"
                    fi
                else
                    if [ "$text_encoder_option" = "1" ]; then
                        config_name="${dataset}_${checkpoint_suffix}_with_text_encoder"
                    else
                        config_name="${dataset}_${checkpoint_suffix}"
                    fi
                fi
                
                generated_dir="${OUTPUT_DIR}/${config_name}"
                metrics_output="${METRICS_DIR}/${model_type}-${config_name}.txt"
                
                echo "Evaluating: $model_type - $config_name"
                

                if [ ! -d "$real_dir" ] || [ ! -d "$generated_dir" ]; then
                    echo "Skipping: Directory not found"
                    continue
                fi
                

                if python evaluate_gen.py \
                    --real_dir="$real_dir" \
                    --generated_dir="$generated_dir" \
                    --output_file="$metrics_output"; then
                    

                    if [ -f "$metrics_output" ]; then
                        fid=$(grep "fid:" "$metrics_output" | cut -d' ' -f2)
                        clip_iqa=$(grep "clip_iqa:" "$metrics_output" | cut -d' ' -f2)
                        echo "$model_type,$dataset,$checkpoint_suffix,$text_encoder_option,$fid,$clip_iqa" >> "${METRICS_DIR}/metrics_summary.csv"
                    fi
                fi
            done
        done
    done
done

python << END
import pandas as pd

try:
    df = pd.read_csv("${METRICS_DIR}/metrics_summary.csv")
    
    summary = df.groupby(['model_type', 'dataset', 'checkpoint_type']).agg({
        'fid': 'mean',
        'clip_iqa': 'mean'
    }).round(4)
    
    print("\nSummary Statistics:")
    print(summary)

except Exception as e:
    print(f"Error generating summary: {e}")
END

echo "Evaluation complete! Results saved in ${METRICS_DIR}/metrics_summary.csv"