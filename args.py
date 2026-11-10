import argparse


def get_args():
    p = argparse.ArgumentParser(description="Knee X-ray segmentation with U-Net")

    p.add_argument("--mode", choices=["train", "eval"], default="train")

    # data
    p.add_argument("--csv_dir", default="data/CSVs")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=2)

    # training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--base_ch", type=int, default=32, help="channels in the first U-Net block")
    p.add_argument("--patience", type=int, default=15, help="early stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)

    # outputs
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--run_name", default="main",
                   help="sub-folder of out_dir; use e.g. 'demo' for the short 5-epoch run")
    p.add_argument("--checkpoint", default="outputs/main/best_model.pth",
                   help="model used in eval mode")
    p.add_argument("--threshold", type=float, default=0.5)

    return p.parse_args()
