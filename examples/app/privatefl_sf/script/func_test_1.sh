echo "==> Start Test"
python ../FedAverage.py --data='mnist' --nclient=2 --nclass=10 --ncpc=5 --model='mnist_fully_connected_IN' --mode='LDP' --round=1 --epsilon=2 --sr=1 --lr=0.1
echo "==> Test Finished!"