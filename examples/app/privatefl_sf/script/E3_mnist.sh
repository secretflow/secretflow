start=$(date +%s)
cd .. &&
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='CDP' --round=150 --epsilon=8 --sr=1 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=3
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=4 --model='mnist_fully_connected_IN' --mode='CDP' --round=150 --epsilon=8 --sr=1 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=3
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=6 --model='mnist_fully_connected_IN' --mode='CDP' --round=150 --epsilon=8 --sr=1 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=3
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=8 --model='mnist_fully_connected_IN' --mode='CDP' --round=150 --epsilon=8 --sr=1 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=3
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=10 --model='mnist_fully_connected_IN' --mode='CDP' --round=150 --epsilon=8 --sr=1 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=3

python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='LDP' --round=150 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=4 --model='mnist_fully_connected_IN' --mode='LDP' --round=150 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=6 --model='mnist_fully_connected_IN' --mode='LDP' --round=150 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=8 --model='mnist_fully_connected_IN' --mode='LDP' --round=150 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=10 --model='mnist_fully_connected_IN' --mode='LDP' --round=150 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3

python log/show.py --E=3 --data='mnist'
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.


