import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch.multiprocessing
from pytorch_lightning import callbacks

from animeganv2.animeganv2 import AnimeGanV2
from animeganv2.data import AnimeGanDataModule

# Fix for "OSError: Too many open files."
torch.multiprocessing.set_sharing_strategy("file_system")


def get_checkpoint_callback(output_path: str):
    checkpoint_dir_path = os.path.join(
        output_path, "checkpoints", datetime.utcnow().strftime(r"%Y%m%d-%H%M%S")
    )
    os.makedirs(checkpoint_dir_path)
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename="checkpoint_{epoch:03d}-{g_loss:.2f}",
        save_last=True,
    )
    return checkpoint_callback


def main(args):
    data = AnimeGanDataModule(
        args.real_input_path,
        args.style_input_path,
        args.batch_size,
        args.val_batch_size,
    )

    model = AnimeGanV2(args.init_epochs)

    trainer = pl.Trainer(
        default_root_dir=args.output_path,
        max_epochs=101,
        gpus=True,
        auto_select_gpus=True,
        benchmark=True,
        multiple_trainloader_mode="max_size_cycle",
        callbacks=[
            callbacks.RichModelSummary(max_depth=2),
            get_checkpoint_callback(args.output_path),
        ],
        detect_anomaly=True,
        fast_dev_run=args.debug,
    )
    trainer.fit(model, data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("real_input_path", type=str)
    parser.add_argument("style_input_path", type=str)
    parser.add_argument("output_path", type=str)
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
