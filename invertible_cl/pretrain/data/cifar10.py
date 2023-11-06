from pl_bolts.datamodules import CIFAR10DataModule

from .transforms import SimCLRViews


class CIFAR10(CIFAR10DataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 256,
            num_workers: int = 8,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            val_split=1000,
            num_workers=num_workers,
            normalize=True,
            batch_size=batch_size
        )

        self._train_transforms = SimCLRViews(
            size=32,
            jitter_strength=0.5,
            blur=False,
            final_transforms=self.default_transforms()
        )
