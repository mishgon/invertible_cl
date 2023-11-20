from typing import Any

from pl_bolts.datamodules import CIFAR10DataModule

from .transforms import SimCLRViews


class CIFAR10(CIFAR10DataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 256,
            num_workers: int = 8,
            **simclr_views_params: Any
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            val_split=1000,
            num_workers=num_workers,
            normalize=True,
            batch_size=batch_size
        )

        params = dict(
            size=32,
            jitter_strength=0.5,
            blur=False,
        )
        params.update(simclr_views_params)
        self._train_transforms = SimCLRViews(
            **params,
            final_transforms=self.default_transforms()
        )
