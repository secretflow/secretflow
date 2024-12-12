from beir import util
import os

# datasets = nq msmarco hotpotqa fiqa trec-covid nfcorpus arguana quora scidocs fever scifact
# Download and save dataset
datasets = ['nq', 
            'msmarco', 
            'hotpotqa']
for dataset in datasets:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(os.getcwd(), "./")
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)

os.system('rm ./*.zip')