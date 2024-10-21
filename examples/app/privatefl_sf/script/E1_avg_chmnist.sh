start=$(date +%s)
cd .. &&
python FedAverage.py --data='chmnist' --nclient=40 --nclass=8 --ncpc=2 --model='alexnet' --mode='CDP' --round=50 --epsilon=2 --sr=0.8 --lr=5e-5 --flr=1e-1 --physical_bs=3 --E=1
python FedAverage.py --data='chmnist' --nclient=40 --nclass=8 --ncpc=2 --model='alexnet' --mode='CDP' --round=120 --epsilon=4 --sr=0.8 --lr=1e-4 --flr=1e-1 --physical_bs=3 --E=1
python FedAverage.py --data='chmnist' --nclient=40 --nclass=8 --ncpc=2 --model='alexnet' --mode='CDP' --round=150 --epsilon=6 --sr=0.8 --lr=1e-4 --flr=1e-1 --physical_bs=3 --E=1
python FedAverage.py --data='chmnist' --nclient=40 --nclass=8 --ncpc=2 --model='alexnet' --mode='CDP' --round=150 --epsilon=8 --sr=0.8 --lr=1e-4 --flr=1e-1 --physical_bs=3 --E=1

python FedAverage.py --data='chmnist' --nclient=20 --nclass=8 --ncpc=2 --model='alexnet' --mode='LDP' --round=15 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=2 --E=1
python FedAverage.py --data='chmnist' --nclient=20 --nclass=8 --ncpc=2 --model='alexnet' --mode='LDP' --round=18 --epsilon=4 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=2 --E=1
python FedAverage.py --data='chmnist' --nclient=20 --nclass=8 --ncpc=2 --model='alexnet' --mode='LDP' --round=22 --epsilon=6 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=2 --E=1
python FedAverage.py --data='chmnist' --nclient=20 --nclass=8 --ncpc=2 --model='alexnet' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=2 --E=1


python log/show.py --E=1 --data='chmnist' --base=1
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.


