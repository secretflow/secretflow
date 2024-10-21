start=$(date +%s)
cd ../transfer &&

python FedTransfer.py --data='cifar10' --nclient=50 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=4 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=4 --bs=64
python FedTransfer.py --data='cifar10' --nclient=200 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=4 --bs=64
python FedTransfer.py --data='cifar10' --nclient=500 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=25 --epsilon=8 --sr=1 --lr=1e-2 --flr=1e-1 --physical_bs=64 --E=4 --bs=64


python FedTransfer.py --data='cifar10' --nclient=50 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=4 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=4 --bs=64
python FedTransfer.py --data='cifar10' --nclient=200 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=4 --bs=64
python FedTransfer.py --data='cifar10' --nclient=500 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=40 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=4 --bs=64


cd .. &&
python log/show.py --E=4 --data='cifar10'
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.