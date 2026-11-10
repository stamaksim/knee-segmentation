from args import get_args
from dataset import get_loader
from evaluate import run_evaluation
from model import UNet
from trainer import Trainer
from utils import get_device, set_seed


def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()

    if args.mode == "train":
        train_loader = get_loader(args.csv_dir, "train", args.batch_size, train=True,
                                  num_workers=args.num_workers, img_size=args.img_size)
        val_loader = get_loader(args.csv_dir, "val", args.batch_size, train=False,
                                num_workers=args.num_workers, img_size=args.img_size)
        model = UNet(base=args.base_ch)
        Trainer(model, train_loader, val_loader, args, device).fit()
    else:
        run_evaluation(args, device)


if __name__ == "__main__":
    main()
