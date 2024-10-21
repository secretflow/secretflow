start=$(date +%s)
cd .. &&
python FedAverage.py --data='mnist' --nclient=50 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='CDP' --round=60 --epsilon=8 --sr=1 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=4
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='CDP' --round=150 --epsilon=8 --sr=1 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=4
python FedAverage.py --data='mnist' --nclient=200 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='CDP' --round=200 --epsilon=8 --sr=0.3 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=4
python FedAverage.py --data='mnist' --nclient=500 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='CDP' --round=300 --epsilon=8 --sr=0.3 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=4

python FedAverage.py --data='mnist' --nclient=50 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='LDP' --round=60 --epsilon=8 --sr=1 --lr=5e-1 --flr=1e-1 --physical_bs=64 --E=4
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='LDP' --round=150 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=4
python FedAverage.py --data='mnist' --nclient=200 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='LDP' --round=200 --epsilon=8 --sr=0.3 --lr=1e-1 --flr=1e-1 --physical_bs=32 --E=4
python FedAverage.py --data='mnist' --nclient=500 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='LDP' --round=300 --epsilon=8 --sr=0.3 --lr=1e-1 --flr=1e-1 --physical_bs=16 --E=4

python log/show.py --E=4 --data='mnist'
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.