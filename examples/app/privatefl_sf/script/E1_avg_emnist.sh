start=$(date +%s)
cd .. &&

python FedAverage.py --data='emnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected' --mode='CDP' --round=20 --epsilon=2 --sr=0.3 --lr=5e-3 --flr=5e-2 --physical_bs=8 --E=1
python FedAverage.py --data='emnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected' --mode='CDP' --round=50 --epsilon=4 --sr=0.3 --lr=5e-3 --flr=1e-1 --physical_bs=8 --E=1
python FedAverage.py --data='emnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected' --mode='CDP' --round=60 --epsilon=6 --sr=0.3 --lr=5e-3 --flr=1e-1 --physical_bs=8 --E=1
python FedAverage.py --data='emnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected' --mode='CDP' --round=100 --epsilon=8 --sr=0.3 --lr=5e-3 --flr=1e-1 --physical_bs=8 --E=1

python FedAverage.py --data='emnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected' --mode='LDP' --round=20 --epsilon=2 --sr=0.3 --lr=5e-2 --flr=1e-1 --physical_bs=64 --E=1
python FedAverage.py --data='emnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected' --mode='LDP' --round=50 --epsilon=4 --sr=0.3 --lr=5e-2 --flr=1e-1 --physical_bs=64 --E=1
python FedAverage.py --data='emnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected' --mode='LDP' --round=60 --epsilon=6 --sr=0.3 --lr=5e-2 --flr=1e-1 --physical_bs=64 --E=1
python FedAverage.py --data='emnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected' --mode='LDP' --round=100 --epsilon=8 --sr=0.3 --lr=5e-2 --flr=1e-1 --physical_bs=64 --E=1

python log/show.py --E=1 --data='emnist' --base=1
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.