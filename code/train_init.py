from lib.model.sdf_init import SDF_Init
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob

@hydra.main(config_path="confs", config_name="base_init")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-{loss}",
        save_on_train_epoch_end=True,
        every_n_epochs=1)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}")

    trainer = pl.Trainer(
        gpus=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=200,
        check_val_every_n_epoch=200,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    model = SDF_Init(opt)
    trainset = create_dataset(opt.dataset.train)
    validset = create_dataset(opt.dataset.valid)
    if opt.model.is_continue == True:
        checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
        # model = IDR.load_from_checkpoint(checkpoint, opt=opt)
        trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
    else: 
        # model = IDR(opt)
        trainer.fit(model, trainset, validset)
    # trainer.fit(model, trainset, validset)


if __name__ == '__main__':
    main()