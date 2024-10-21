start=$(date +%s)
cd ../transfer &&

python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='CDP' --round=12 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='resnext' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-2 --flr=1e-1 --physical_bs=64 --E=2 --bs=64

python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=15 --epsilon=6 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='CDP' --round=8 --epsilon=6 --sr=1 --lr=1e-2 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='resnext' --model='linear_model_DN_IN' --mode='CDP' --round=15 --epsilon=6 --sr=1 --lr=1e-2 --flr=1e-1 --physical_bs=64 --E=2 --bs=64

python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=10 --epsilon=4 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='CDP' --round=5 --epsilon=4 --sr=1 --lr=1e-2 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='resnext' --model='linear_model_DN_IN' --mode='CDP' --round=10 --epsilon=4 --sr=1 --lr=1e-2 --flr=1e-1 --physical_bs=64 --E=2 --bs=64

python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=5 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='CDP' --round=5 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='resnext' --model='linear_model_DN_IN' --mode='CDP' --round=5 --epsilon=2 --sr=1 --lr=1e-2 --flr=1e-1 --physical_bs=64 --E=2 --bs=64



python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=25 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='resnext' --model='linear_model_DN_IN' --mode='LDP' --round=40 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64

python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=6 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=15 --epsilon=6 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='resnext' --model='linear_model_DN_IN' --mode='LDP' --round=40 --epsilon=6 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64

python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=15 --epsilon=4 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=8 --epsilon=4 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='resnext' --model='linear_model_DN_IN' --mode='LDP' --round=15 --epsilon=4 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64

python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=10 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=3 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='resnext' --model='linear_model_DN_IN' --mode='LDP' --round=10 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64


cd .. &&
python log/show.py --E=2 --data='cifar10'
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.