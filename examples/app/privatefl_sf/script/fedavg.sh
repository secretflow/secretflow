cd .. &&
python FedAverage.py --data='mnist' --nclient=50 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='CDP' --round=60 --epsilon=2 --sr=1 --lr=5e-3 --flr=1e-2 --physical_bs=64 --E=1