start=$(date +%s)
cd .. &&

python FedAverage.py --data='purchase' --nclient=50 --nclass=100 --ncpc=2 --model='purchase_fully_connected' --mode='LDP' --round=200 --epsilon=2 --sr=0.1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=1
python FedAverage.py --data='purchase' --nclient=50 --nclass=100 --ncpc=2 --model='purchase_fully_connected' --mode='LDP' --round=350 --epsilon=4 --sr=0.1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=1
python FedAverage.py --data='purchase' --nclient=50 --nclass=100 --ncpc=2 --model='purchase_fully_connected' --mode='LDP' --round=410 --epsilon=6 --sr=0.1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=1
python FedAverage.py --data='purchase' --nclient=50 --nclass=100 --ncpc=2 --model='purchase_fully_connected' --mode='LDP' --round=420 --epsilon=8 --sr=0.1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=1

python FedAverage.py --data='purchase' --nclient=50 --nclass=100 --ncpc=2 --model='purchase_fully_connected' --mode='CDP' --round=50 --epsilon=2 --sr=0.1 --lr=1e-3 --flr=1e-3 --physical_bs=64 --E=1
python FedAverage.py --data='purchase' --nclient=50 --nclass=100 --ncpc=2 --model='purchase_fully_connected' --mode='CDP' --round=80 --epsilon=4 --sr=0.1 --lr=1e-3 --flr=1e-3 --physical_bs=64 --E=1
python FedAverage.py --data='purchase' --nclient=50 --nclass=100 --ncpc=2 --model='purchase_fully_connected' --mode='CDP' --round=100 --epsilon=6 --sr=0.1 --lr=1e-3 --flr=1e-3 --physical_bs=64 --E=1
python FedAverage.py --data='purchase' --nclient=50 --nclass=100 --ncpc=2 --model='purchase_fully_connected' --mode='CDP' --round=150 --epsilon=8 --sr=0.1 --lr=1e-3 --flr=1e-3 --physical_bs=64 --E=1

python log/show.py --E=1 --data='purchase' --base=1
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.