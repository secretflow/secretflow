import os
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist','cifar10','cifar100','fashionmnist','emnist','purchase','chmnist'])
    parser.add_argument('--E',  type=int, default=1,
                        help='the index of experiment in AE')
    parser.add_argument('--base',  type=int, default=0,
                        help='Baseline (Fedavg)')
    args = parser.parse_args()
    return args

args = parse_arguments()

E = args.E
data = args.data
base = args.base

if E == 1:
    results_df = pd.DataFrame(columns=["data","mode","epsilon","accuracy"])
    directory = f'log/E{E}'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
               path = os.path.join(root, file)
               if base == 1:
                   if data in path.split('_')[0].split('/') and 'IN' not in path:
                       df = pd.read_csv(path, header=None)
                       new_header = df.iloc[0]
                       df = df[1:]
                       df.columns = new_header
                       results_df = results_df._append(
                           {"data": df["data"].values[0],"mode": df["mode"].values[0],
                            "epsilon": df["epsilon"].values[0], "accuracy": df["accuracy"].values[0]},
                           ignore_index=True)
               else:
                   if data in path.split('_')[0].split('/') and 'IN' in path:
                       df = pd.read_csv(path, header=None)
                       new_header = df.iloc[0]
                       df = df[1:]
                       df.columns = new_header
                       results_df = results_df._append(
                           {"data": df["data"].values[0],"mode": df["mode"].values[0],
                            "epsilon": df["epsilon"].values[0], "accuracy": df["accuracy"].values[0]},
                           ignore_index=True)
    print(f'==> The results for {data} in E{E} is:')
    print(results_df.sort_values(by = ['mode', 'epsilon']).reset_index(drop=True))


elif E == 2:
    results_df = pd.DataFrame(columns=["data","mode","model","epsilon","accuracy"])
    directory = f'log/E{E}'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
               path = os.path.join(root, file)
               if data in path.split('_')[0].split('/'):
                   df = pd.read_csv(path, header=None)
                   new_header = df.iloc[0]
                   df = df[1:]
                   df.columns = new_header
                   results_df = results_df._append(
                       {"data": df["data"].values[0],"mode": df["mode"].values[0],
                        "model": df["model"].values[0], "epsilon": df["epsilon"].values[0], "accuracy": df["accuracy"].values[0]},
                       ignore_index=True)
    print(f'==> The results for {data} in E{E} is:')
    print(results_df.sort_values(by =  ['model', 'mode', 'epsilon']).reset_index(drop=True))

elif E == 3:
    results_df = pd.DataFrame(columns=["data", "mode", "ncpc", "accuracy"])
    directory = f'log/E{E}'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                if data in path:
                    df = pd.read_csv(path, header=None)
                    new_header = df.iloc[0]
                    df = df[1:]
                    df.columns = new_header
                    results_df = results_df._append(
                        {"data": df["data"].values[0], "mode": df["mode"].values[0],
                         "ncpc": df["ncpc"].values[0], "accuracy": df["accuracy"].values[0]},
                        ignore_index=True)
    print(f'==> The results for {data} in E{E} is:')
    print(results_df.sort_values(by = ['mode', "ncpc"]).reset_index(drop=True))

else:
    results_df = pd.DataFrame(columns=["data", "mode", "nc", "accuracy"])
    directory = f'log/E{E}'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                if data in path:
                    df = pd.read_csv(path, header=None)
                    new_header = df.iloc[0]
                    df = df[1:]
                    df.columns = new_header
                    results_df = results_df._append(
                        {"data": df["data"].values[0], "mode": df["mode"].values[0],
                         "nc": df["num_client"].values[0], "accuracy": df["accuracy"].values[0]},
                        ignore_index=True)
    print(f'==> The results for {data} in E{E} is:')
    print(results_df.sort_values(by = ['mode', "nc"]).reset_index(drop=True))



