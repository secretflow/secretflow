import sys
sys.path.append('../')
from datasets import *
import os
from pl_bolts.models.self_supervised import SimCLR
from resnext import resnext
from CLIP import clip
from tqdm import tqdm
from collections import OrderedDict

def extract(DATA_NAME, NUM_CLIENTS, NUM_CLASSES, NUM_CLASES_PER_CLIENT, ENCODER, BATCH_SIZE,  preprocess = None, path = None):
    if ENCODER == "simclr":
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
        model = simclr.encoder
        model.eval()
    elif ENCODER == "resnext":
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize(32), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        model = resnext(cardinality=8, num_classes=100, depth=29, widen_factor=4, dropRate=0, )
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        checkpoint = torch.load("model/model_best.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        model.module.classifier = torch.nn.Identity()
    elif ENCODER == "clip":
        model, preprocess = clip.load('ViT-B/32', 'cpu')
        model = model.encode_image


    train_dataloaders, test_dataloaders = gen_random_loaders(DATA_NAME, '~/torch_data', NUM_CLIENTS,
                                                             BATCH_SIZE, NUM_CLASES_PER_CLIENT, NUM_CLASSES, preprocess)
    os.makedirs(f'{path}/', exist_ok=True)
    print('=====> model loaded')
    print('=====> start extracting')

    for index in tqdm(range(NUM_CLIENTS)):
        features_train = []
        y_train = []
        for x, y in train_dataloaders[index]:
            features = model(x)
            try:
                features_train.append(features.cpu().detach().numpy())
            except:
                features_train.append(features[0].cpu().detach().numpy())
            y_train.append(y)
        features_train = np.concatenate(features_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        np.save(f'{path}/{index}_train_x.npy', features_train)
        np.save(f'{path}/{index}_train_y.npy', y_train)
        features_test = []
        y_test = []
        for x, y in test_dataloaders[index]:
            features = model(x)
            try:
                features_test.append(features.cpu().detach().numpy())
            except:
                features_test.append(features[0].cpu().detach().numpy())
            y_test.append(y)
        features_test = np.concatenate(features_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        np.save(f'{path}/{index}_test_x.npy', features_test)
        np.save(f'{path}/{index}_test_y.npy', y_test)
    print('=====> finish extracting')