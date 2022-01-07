import pytorch_lightning as pl
from pytorch_lightning import callbacks

from animeganv2.animeganv2 import AnimeGanV2
from animeganv2.data import AnimeGanDataModule


def main(args):
    data = AnimeGanDataModule(
        "/mnt/d/datasets/AnimeGANDataset/train_photo/",
        "/mnt/d/datasets/AnimeGANDataset/Shinkai/",
        2,
    )

    model = AnimeGanV2()

    trainer = pl.Trainer(
        max_epochs=101,
        gpus=True,
        auto_select_gpus=True,
        benchmark=True,
        multiple_trainloader_mode="max_size_cycle",
        callbacks=[callbacks.ModelSummary(max_depth=2), callbacks.RichProgressBar()],
        detect_anomaly=True,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main(None)
