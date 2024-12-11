import os
import glob
import pandas as pd
import re

def extract_parameters_from_path(path):
    """Extract experiment parameters from the output path."""
    params = {
        "model": "unknown",
        "dataset": "unknown",
        "p_value": "unknown",
        "s_value": "unknown",
        "protection": "unknown"
    }
    
    # Extract model version (sd1.5 or sd2)
    if "sdv1-5" in path:
        params["model"] = "sd1.5"
    elif "sdv2" in path:
        params["model"] = "sd2"
    
    # Extract dataset
    if "celeba" in path:
        params["dataset"] = "celeba"
    elif "mscoco" in path:
        params["dataset"] = "mscoco"
    
    # Extract p value
    p_match = re.search(r'p(\d+\.\d+)', path)
    if p_match:
        params["p_value"] = p_match.group(1)
    
    # Extract s value if present
    s_match = re.search(r's(\d+\.\d+)', path)
    if s_match:
        params["s_value"] = s_match.group(1)
    
    # Determine protection type
    if "wanet" in path:
        if "unconditional" in path:
            params["protection"] = "protected_unconditional"
        else:
            params["protection"] = "protected_conditional"
    else:
        params["protection"] = "unprotected"
    
    return params

def extract_metrics_from_file(file_path):
    """Extract metrics from result file."""
    metrics = {
        "memorization_strength": None,
        "model_status": None,
        "fid": None
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Extract Memorization Strength
            mem_match = re.search(r'Memorization Strength: ([\d.]+)', content)
            if mem_match:
                metrics["memorization_strength"] = float(mem_match.group(1))
            
            # Extract Model Status
            status_match = re.search(r'Model Status: (.+)', content)
            if status_match:
                metrics["model_status"] = status_match.group(1).strip()
            
            # Extract FID
            fid_match = re.search(r'FID: ([\d.]+)', content)
            if fid_match:
                metrics["fid"] = float(fid_match.group(1))
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return metrics

def main():
    # Initialize list to store all results
    results = []
    
    # Search for all output directories
    output_dirs = glob.glob("output/*")
    print(f"Found {len(output_dirs)} output directories")
    
    for output_dir in output_dirs:
        result_file = os.path.join(output_dir, "result.txt")
        print(f"Processing: {output_dir}")
        
        if os.path.exists(result_file):
            # Extract parameters from path
            params = extract_parameters_from_path(output_dir)
            
            # Extract metrics from result file
            metrics = extract_metrics_from_file(result_file)
            
            # Combine parameters and metrics
            result_row = {
                **params,
                **metrics,
                "output_path": output_dir
            }
            
            print(f"Extracted data: {result_row}")
            results.append(result_row)
        else:
            print(f"No result file found in {output_dir}")
    
    if not results:
        print("No results were collected. Check if the output directories and result files exist.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    print("\nDataFrame columns:", df.columns.tolist())
    
    # Save to CSV
    csv_path = "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total experiments processed: {len(results)}")
    if len(results) > 0:
        print(f"Unique models: {df['model'].nunique()}")
        print(f"Unique datasets: {df['dataset'].nunique()}")
        if 'fid' in df.columns and df['fid'].notna().any():
            print(f"Average FID: {df['fid'].mean():.2f}")

if __name__ == "__main__":
    main()