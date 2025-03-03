import argparse
import itertools
import os
import shutil
from itertools import product as product
from math import sqrt as sqrt

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from PIL import ImageFile

import train_model
from model import Conv3dAutoencoder
from training_sub import weights_init


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of CAE")
    parser.add_argument(
        "--training_validation_path", metavar="DIR", help="training_validation_path", default="/dataset/spitzer_data/"
    )
    parser.add_argument("--savedir_path", metavar="DIR", default="/workspace/weights/search/", help="savedire path")
    # minibatch
    parser.add_argument("--num_epoch", type=int, default=300, help="number of total epochs to run (default: 300)")
    parser.add_argument("--train_mini_batch", default=32, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("--val_mini_batch", default=128, type=int, help="Validation mini-batch size (default: 128)")
    # random seed
    parser.add_argument("--random_state", "-r", type=int, default=123)
    # 学習率
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    # option
    parser.add_argument("--wandb_project", type=str, default="demo")
    parser.add_argument("--wandb_name", type=str, default="demo1")

    return parser.parse_args()


# Training of SSD
def main(args):
    """Train CAE.

    :Example command:
    >>> python train_main.py /dataset/spitzer_data --savedir_path /workspace/webdataset_weights/Ring_selection_compare/ \
        --NonRing_data_path /workspace/NonRing_png/region_NonRing_png/ \
        --validation_data_path /workspace/cut_val_png/region_val_png/ \
        -s -i 0 --NonRing_remove_class_list 3 --Ring_mini_batch 16 --NonRing_mini_batch 2 --Val_mini_batch 64 \
        --l18_infer --ring_select

    """
    torch.manual_seed(args.random_state)
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.savedir_path):
        print("REMOVE FILES...")
        shutil.rmtree(args.savedir_path)
    os.makedirs(args.savedir_path, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "fits_random_state": args.random_state,
            "train_mini_batch": args.train_mini_batch,
            "val_mini_batch": args.val_mini_batch,
        },
    )

    model = Conv3dAutoencoder()
    model.apply(weights_init)
    wandb.watch(model, log_freq=100)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False
    )

    train_model_params = {
        "net": model,
        "criterion": nn.MSELoss(),
        "optimizer": optimizer,
        "num_epochs": args.num_epoch,
        "args": args,
        "device": device,
        "run": run,
    }

    train_model.train_model(**train_model_params)

    artifact = wandb.Artifact("training_log", type="dir")
    artifact.add_dir(args.savedir_path)
    run.log_artifact(artifact, aliases=["latest", "best"])

    run.alert(title="学習が終了しました", text="学習が終了しました")
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
