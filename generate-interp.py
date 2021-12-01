# Usage (How to generate interpolation movie):
#   python generate-interp.py --size 512 --pics 5 --steps 30 --ckpt checkpoint/009200.pt
#   ffmpeg -r 25 -i sample-interp/%06d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p sample-interp-9200.mp4

import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

def interpolate(zs, steps):
   out = []
   for i in range(len(zs)-1):
    for index in range(steps):
     fraction = index/float(steps) 
     out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
   return out

def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()

        base_zs = []
        for i in tqdm(range(args.pics)):
          base_zs.append(torch.randn(args.sample, args.latent, device=device))

        zs = interpolate(base_zs, args.steps)
        print(f"args.pics: {args.pics}")
        print(f"args.steps: {args.steps}")
        print(f"len of frames: {len(zs)}")
        #print(f"len of base_zs: {len(base_zs)}")

        for i in tqdm(range(len(zs))):
            # print(f"sample-interp/{str(i).zfill(6)}.png")
            sample_z = zs[i]

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"sample-interp/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of base images"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="number of interpolation steps"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
