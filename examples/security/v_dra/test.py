import torch
from data_reconstruction_attack import *

import argparse
import os
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from model_resnet import ResNet18
from model_stylegan2 import Generator


def main(args):
    torch.manual_seed(args.random_seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.gpu_id if use_cuda else 'cpu')

    cuda_kwargs = {}
    if use_cuda:
        torch.cuda.set_device(args.gpu_id)
        cudnn.benchmark = True
        cuda_kwargs = {'num_workers': args.workers, 'pin_memory': True}

    # load test data
    if not os.path.exists(args.data_path):
        raise Exception("Test Dataset does not exist!")
    test_dataset = torch.load(args.data_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, **cuda_kwargs
    )

    print("Test dataset length:", len(test_dataset))

    # load target model
    target_model = ResNet18()
    if not os.path.exists(args.trg_model_path):
        raise Exception("Target Model does not exist!")
    target_model = torch.load(args.trg_model_path)
    target_model = target_model.to(device)
    target_model.eval()
    print("\nTarget model:", target_model, "\n")

    # load attack model
    size = 64
    style_dim = 512
    n_mlp = 8
    channel_multiplier = 1
    attack_model = Generator(
        size, style_dim, n_mlp, channel_multiplier=channel_multiplier
    )
    if not os.path.exists(args.att_model_path):
        raise Exception("Attack Model does not exist!")
    attack_model = torch.load(args.att_model_path)
    attack_model = attack_model.to(device)
    attack_model.eval()

    # data reconstruction attacker
    attacker = DataReconstructionAttacker(
        device, target_model, attack_model, iter_z=1, iter_w=2000
    )

    # denorm function for image tensor
    denorm = DeNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    recon_iter = iter(test_loader)
    for i in range(args.recon_num):
        # save ground truth image
        ground_truth, _ = next(recon_iter)
        img = denorm(ground_truth.detach())
        vutils.save_image(img, 'tmp/truth_{}.png'.format(i), normalize=False)
        ground_truth = ground_truth.to(device)

        # get target feature
        with torch.no_grad():
            target_feature = target_model.client_model(ground_truth)

        # get reconstruction result
        attack_result = attacker.attack(target_feature)
        vutils.save_image(attack_result, 'tmp/recon_{}.png'.format(i), normalize=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Example of Enhanced Data Reconstruction Attack using Gan"
    )
    # gpu setting
    parser.add_argument('--gpu_id', type=int, default=0, help="which gpu id to run")
    parser.add_argument('--workers', type=int, default=8, help="number of workers")
    # attack setting
    parser.add_argument(
        '--random_seed', type=int, default=666, help="random seed for the experiment"
    )
    parser.add_argument(
        '--trg_model_path',
        type=str,
        default='tmp/target_model.pt',
        help="the path of target model",
    )
    parser.add_argument(
        '--att_model_path',
        type=str,
        default='tmp/attack_model.pt',
        help="the path of attack model",
    )
    parser.add_argument(
        '--data_path', type=str, default='tmp/data.npz', help="the path of test data"
    )
    parser.add_argument(
        '--recon_num', type=int, default=1, help="number of reconstructed images"
    )
    args = parser.parse_args()

    main(args)
