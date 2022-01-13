import argparse

import pytorch_lightning as pl
from pytorch_lightning import callbacks

from animeganv2.animeganv2 import AnimeGanV2
from animeganv2.data import AnimeGanDataModule


def main(args):
    data = AnimeGanDataModule(
        "/mnt/d/datasets/AnimeGANDataset/train_photo/",
        "/mnt/d/datasets/AnimeGANDataset/Shinkai/",
        args.batch_size,
        args.val_batch_size,
    )

    model = AnimeGanV2(args.init_epochs)

    trainer = pl.Trainer(
        max_epochs=101,
        gpus=True,
        auto_select_gpus=True,
        benchmark=True,
        multiple_trainloader_mode="max_size_cycle",
        callbacks=[callbacks.ModelSummary(max_depth=2), callbacks.RichProgressBar()],
        detect_anomaly=True,
        fast_dev_run=args.debug,
    )
    trainer.fit(model, data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--val-batch-size", "-vb", type=int, default=4)
    parser.add_argument("--init-epochs", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Set init epochs to 0 if in debug mode to fast_dev_run through
    # all losses and models etc.
    if args.debug:
        args.init_epochs = 0

    main(args)
