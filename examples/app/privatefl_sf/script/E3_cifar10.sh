start=$(date +%s)
cd ../transfer &&

python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=4 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=6 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=8 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=10 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3 --bs=64

python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=4 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=6 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=8 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=10 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=3 --bs=64

cd .. &&
python log/show.py --E=3 --data='cifar10'
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.