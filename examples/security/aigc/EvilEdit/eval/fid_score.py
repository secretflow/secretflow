from cleanfid import fid
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args):
    score = fid.compute_fid(args.fdir1, args.fdir2, device=args.device)
    print(f'FID Score = {score}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FID Score')
    parser.add_argument('--fdir1', type=str, required=True)
    parser.add_argument('--fdir2', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)