start=$(date +%s)
cd ../transfer &&

python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=100 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=40 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=35 --epsilon=8 --sr=1 --lr=1 --flr=1e-1 --physical_bs=64 --E=2 --bs=256
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=28 --epsilon=2 --sr=1 --lr=1 --flr=1e-1 --physical_bs=64 --E=2 --bs=256

python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=70 --epsilon=6 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=60 --epsilon=4 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=32 --epsilon=6 --sr=1 --lr=1 --flr=1e-1 --physical_bs=64 --E=2 --bs=256
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=29 --epsilon=4 --sr=1 --lr=1 --flr=1e-1 --physical_bs=64 --E=2 --bs=256


python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='CDP' --round=30 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='CDP' --round=30 --epsilon=2 --sr=1 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=9 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=9 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64

python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='CDP' --round=35 --epsilon=6 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='CDP' --round=35 --epsilon=4 --sr=1 --lr=5e-3 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=9 --epsilon=6 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=9 --epsilon=4 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=64 --E=2 --bs=64


cd .. &&
python log/show.py --E=2 --data='cifar100'
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.