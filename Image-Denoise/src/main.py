import argparse
import torch
import matplotlib.pyplot as plt
import os
from data import NoisyBSDSDataset
from argument import parse
from model import DnCNN, UDnCNN, DUDnCNN, DnCNNNoBN
import nntools as nt
from utils import DenoisingStatsManager, plot
from PIL import Image
import numpy as np
import torchvision as tv


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset
    train_set = NoisyBSDSDataset(
        args.root_dir, image_size=args.image_size, sigma=args.sigma)
    test_set = NoisyBSDSDataset(
        args.root_dir, mode='test', image_size=args.test_image_size, sigma=args.sigma)

    # model
    if args.model == 'dncnn':
        net = DnCNN(args.D, C=args.C).to(device)
    elif args.model == 'udncnn':
        net = UDnCNN(args.D, C=args.C).to(device)
    elif args.model == 'dudncnn':
        net = DUDnCNN(args.D, C=args.C).to(device)
    elif args.model == 'dncnnnobn':
        net = DnCNNNoBN(args.D, C=args.C).to(device)
    else:
        raise NameError('Please enter: dncnn, udncnn, dudncnn or dncnnnobn')

    # optimizer
    adam = torch.optim.Adam(net.parameters(), lr=args.lr)

    # stats manager
    stats_manager = DenoisingStatsManager()

    files = os.listdir(os.path.join(args.root_dir,"train"))
    img_path = os.path.join(args.root_dir,"train", files[0])
    clean = Image.open(img_path).convert('RGB')
    #clean.save("Meow.png")
    
    i = np.random.randint(clean.size[0] - args.image_size[0])
    j = np.random.randint(clean.size[1] - args.image_size[1])

    clean = clean.crop([i, j, i+args.image_size[0], j+args.image_size[1]])

    transform = tv.transforms.Compose([
        # convert it to a tensor
        tv.transforms.ToTensor(),
        # normalize it to the range [−1, 1]
        tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    untransform = tv.transforms.Compose([
        # convert it to a tensor
        #tv.transforms.ToTensor(),
        # normalize it to the range [0, 1]
        tv.transforms.Normalize((-0.5,-0.5, -0.5), (2, 2, 2))
    ])

    clean = transform(clean)
    noisy = clean + 2 / 255 * args.sigma * torch.randn(clean.shape)
    
    plt.imshow(torch.clamp(torch.permute(clean,(1,2,0)),min=0,max=1).numpy())
    plt.axis("off")
    #plt.tight_layout()
    plt.savefig("Meow.png",bbox_inches = 'tight')

    noisy = untransform(noisy)
    noisy = torch.permute(noisy,(1,2,0))
    noisy = torch.clamp(noisy,min=0,max=1)
    noisy = noisy.numpy()
    #plt.imshow()
    plt.imshow(noisy)
    plt.axis("off")
    #plt.tight_layout()
    plt.savefig("NoisyMeow.png",bbox_inches = 'tight')
    # experiment
    exp = nt.Experiment(net, train_set, test_set, adam, stats_manager, batch_size=args.batch_size,
                        output_dir=args.output_dir, perform_validation_during_training=True)

    # run
    if args.plot:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 7))
        exp.run(num_epochs=args.num_epochs, plot=lambda exp: plot(exp, fig=fig, axes=axes,
                                                noisy=test_set[73][0]))
    else:
        exp.run(num_epochs=args.num_epochs)


if __name__ == '__main__':
    args = parse()
    print(args)
    run(args)
