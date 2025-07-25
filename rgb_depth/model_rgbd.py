import copy
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from transformers.models.bert.modeling_bert import BertEmbeddings

from model.text import TextEncoder
from model.vision.base_model import JEPA_base
from utils.types import Number

# pylint: disable=pointless-string-statement

BERT_MODEL_NAME: str = "bert-base-uncased"
PRETRAINED_TEXT_ENCODER: bool = True


class IDJEPA(JEPA_base, pl.LightningModule):
    def __init__(
        self,
        decoder_depth: int = 6,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        target_scale_interval: Tuple[float, float] = (0.15, 0.2),
        context_aspect_ratio: Number = 1,
        context_scale: Tuple[float, float] = (0.85, 1.0),
        num_target_blocks: int = 4,  # number of distinct target blocks per image
        m: float = 0.996,  # momentum
        momentum_limits: Tuple[float, float] = (0.996, 1.0),
        testing_purposes_only: bool = False,
        **kwargs,
    ):
        pl.LightningModule.__init__(self)
        JEPA_base.__init__(
            self,
            decoder_depth=decoder_depth,
            num_target_blocks=num_target_blocks,
            **kwargs,
        )
        if not testing_purposes_only:
            self.save_hyperparameters()

        # Define hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m  # momentum
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale_interval = target_scale_interval
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale

        # Optimisation parameters
        self.momentum_limits = momentum_limits
        self.criterion = nn.MSELoss()

    @staticmethod
    def randomly_select_starting_patch_for_block(
        patch_dim: Tuple[int, int], 
        block_dim: Tuple[int, int],
        seed: Optional[int] = None,
    ) -> int:
        """
        Randomly selects the patch defining the 2D block's starting position (on a linear index).

        Parameters:
        patch_dim (Tuple[int, int]): A tuple containing the number of patches in each dimension (width and height).
        block_dim (Tuple[int, int]): A tuple containing the number of patches in each dimension (width and height) of the block from which the patch is to be extracted.
        seed (Optional[int]): An optional random seed for reproducibility.

        Returns:
        int: The starting position of the patch within the block, represented as a linear index.

        NOTE:
        Patches are the basic (processing) units of the image (e.g. 16x16 pixels).
        Blocks are larger regions composed of multiple patches.
        In training, the model attempts to understand blocks within an image - ie. context blocks - tby processing it one patch at a time,
        and uses this understanding is used to predict the structure and content of (the target blocks within) an image in a more abstrac way.

        Linear index coordinates are used to define the starting patch for a block,
        and map 2D pixel coordinates onto a 1D array index (flattened form).
        """
        if seed is not None:
            torch.manual_seed(seed)  # Set the random seed for reproducibility

        def random_int(limit: int) -> int:
            return torch.randint(0, limit, (1,)).item()

        num_patches_h, num_patches_w = (
            patch_dim  # The number of patches in each dimension (width and height)
        )
        num_blocks_h, num_blocks_w = (
            block_dim  # The number of patches in each dimension (width and height)
        )

        max_start_index_h: int = num_patches_h - num_blocks_h + 1 # +1 because of the limit 
        max_start_index_w: int = num_patches_w - num_blocks_w + 1
        assert all(
            (
                num_blocks_h <= num_patches_h,
                num_blocks_w <= num_patches_w,
            )
        ), f"Blocks cannot be smaller than patches along any dimension, but there were more blocks than patches along at least one dimension ({patch_dim=}, {block_dim=})"

        start_index_h: int = random_int(max_start_index_h)
        start_index_w: int = random_int(max_start_index_w)

        # Convert the 2D coordinate to a linear index
        # x1y1, x2y1, x3y1, ...
        # x1y2, x2y2, x3y3, ...
        # ... , ... , ... , ...
        # <--- patch_width --->
        start_index: int = (
            start_index_h
            * num_patches_w  # index of row `start_y` in flattened (1D) form
        ) + start_index_w  # position in row

        return start_index

    @staticmethod
    def generate_target_patches(
        patch_dim: Tuple[int, int],
        aspect_ratio: Number,
        scale: Number,
        num_target_blocks: int,
        seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], Set[int]]:
        """
        Generate (spatial) target patches for each 2D target block.

        Args:
            patch_dim (Tuple[int, int]): The number of patches in each dimension (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for target blocks.
            scale (Number): Scaling factor for the number of patches in the target block.
            num_target_blocks (int): Number of target blocks to generate.
            seed (Optional[int]): An optional random seed for reproducibility.

        Returns:
            Tuple[List[List[int]], Set[int]]:
                - target_patches: A list of lists containing indices of patches for each target block.
                - all_patches: A set of all unique patches used in target blocks.
        """

        # Extract the number of patches in each dimension
        num_patches_h, num_patches_w = patch_dim

        # Calculate the number of patches in the target block
        num_patches_block: int = int(num_patches_h * num_patches_w * scale)

        # Calculate the height and width of the target block maintaining the aspect ratio
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = sqrt(num_patches_block/aspect_ratio)
        """
        num_blocks_h: int = int(
            torch.sqrt(torch.tensor(num_patches_block / aspect_ratio))
        )
        num_blocks_w: int = int(aspect_ratio * num_blocks_h)

        block_dim: Tuple[int, int] = num_blocks_h, num_blocks_w

        # Initialize structures to hold target patches and all unique patches
        target_patches: List[List[int]] = []
        all_patches: Set[int] = set()  # Using a set for fast membership checks

        _target_patches: List[List[int]] = []
        _all_patches: Set[int] = set()  # Using a set for fast membership checks

        # For each of the target blocks to generate
        for target_block_idx in range(num_target_blocks):
            start_patch: int = IJEPA.randomly_select_starting_patch_for_block(
                patch_dim=patch_dim,
                block_dim=block_dim,
                seed=target_block_idx * seed if seed is not None else None,
            )

            # Initialize list to hold the patches for the target block
            patches: List[int] = []
            # Collect patches within the target block
            for h in range(num_blocks_h):
                for w in range(num_blocks_w):
                    patch_start_position: int = start_patch + h * num_patches_w + w

                    patches.append(patch_start_position)

                    # Only updated if the start position is not already present
                    all_patches.add(patch_start_position)

            # Store the patches for the current target block
            target_patches.append(patches)

            # Generate all patch indices in block using tensor operations
            h = torch.arange(num_blocks_h)
            w = torch.arange(num_blocks_w)
            hw_grid = torch.cartesian_prod(
                h, w
            )  # Efficiently generates all combinations of h, w

            block_patch_indices = start_patch + (
                hw_grid[:, 0] * num_patches_w + hw_grid[:, 1]
            )

            _target_patches.append(block_patch_indices.tolist())
            _all_patches.update(block_patch_indices.tolist())

        assert len(target_patches) == len(_target_patches)
        assert len(all_patches) == len(_all_patches)

        assert target_patches == _target_patches
        assert all_patches == _all_patches

        return target_patches, all_patches

    @staticmethod
    def generate_context_patches(
        patch_dim: Tuple[int, int],
        aspect_ratio: Number,
        scale: Number,
        target_patches_to_exclude: Set[int],
        seed: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a list of patch indices for the 2D context block, excluding target patches.

        Args:
            patch_dim (Tuple[int, int]): The number of patches in each dimension (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for the context block.
            scale (Number): Scaling factor for the number of patches in the context block.
            target_patches_to_exclude (Set[int]): Set containing indices of target patches.
            seed (Optional[int]): An optional random seed for reproducibility.

        Returns:
            List[int]: A list of patch indices for the context block excluding target patches.
        """
        # Extract the number of patches in each dimension
        num_patches_h, num_patches_w = patch_dim

        # Calculate the number of patches in the context block
        num_patches_block: int = int(num_patches_h * num_patches_w * scale)

        # Calculate the height and width of the context block maintaining the aspect ratio
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = (num_patches_block/aspect_ratio)**.5
        """
        num_blocks_h: int = int(
            torch.sqrt(torch.tensor(num_patches_block / aspect_ratio))
        )
        num_blocks_w: int = int(aspect_ratio * num_blocks_h)

        block_dim: Tuple[int, int] = num_blocks_h, num_blocks_w

        # Randomly select the starting patch for the context block
        start_patch: int = IJEPA.randomly_select_starting_patch_for_block(
            patch_dim=patch_dim,
            block_dim=block_dim,
            seed=seed,
        )

        context_patches_set: Set[int] = set()
        # Collect patches within the context block
        for h in range(num_blocks_h):
            for w in range(num_blocks_w):
                patch_start_position: int = start_patch + h * num_patches_w + w
                context_patches_set.add(patch_start_position)

        # Distinctive feature
        # Exclude the target patches
        context_patches: List[int] = list(
            context_patches_set.difference(target_patches_to_exclude)
        )

        # Another way to generate
        h = torch.arange(num_blocks_h)
        w = torch.arange(num_blocks_w)
        hw_grid = torch.cartesian_prod(h, w)

        _context_patches_tensor: torch.Tensor = start_patch + (
            +hw_grid[:, 0] * num_patches_w + hw_grid[:, 1]
        )

        _context_patches_set = set(_context_patches_tensor.tolist())

        _context_patches: List[int] = list(
            _context_patches_set.difference(target_patches_to_exclude)
        )

        assert len(context_patches) == len(_context_patches)
        assert context_patches == _context_patches

        return context_patches

    def forward(  # pylint: disable=arguments-differ
        self,
        x_rgb: torch.Tensor,
        x_dep: torch.Tensor,
        target_aspect_ratio: float,
        target_scale: float,
        context_aspect_ratio: Number,
        context_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_patches: List[List[int]]
        all_unique_target_patches: Set[int]

        target_patches, all_unique_target_patches = IDJEPA.generate_target_patches(
            patch_dim=self.patch_embed.patch_shape,  # The number of patches in each dimension
            aspect_ratio=target_aspect_ratio,
            scale=target_scale,
            num_target_blocks=self.num_target_blocks,
        )

        context_patches: List[int] = IDJEPA.generate_context_patches(
            patch_dim=self.patch_embed.patch_shape,
            aspect_ratio=context_aspect_ratio,
            scale=context_scale,
            target_patches_to_exclude=all_unique_target_patches,
        )

        return self.forward_base(
            x_rgb= x_rgb,  # (batch_size, channels, img_height, img_width)
            x_dep= x_dep,
            target_patches=target_patches,
            context_patches=context_patches,
        )  # --> return (prediction_block, actual_target_block)

    def update_momentum(self, m: float) -> None:
        """
        Update the teacher model parameters using momentum.

        Args:
            m (float): Momentum coefficient for the exponential moving average update.
        """
        # Enable eval mode to disable layers like dropout and batch normalization
        student_model: nn.Module = self.encoder.eval()
        teacher_model: nn.Module = self.teacher_encoder.eval()

        """
        Manual parameter updates:
        Manually update the teacher's parameters using a momentum term, ensuring
        that the teacher's parameters are a smoothed version of the student's parameters,
        thus reducing the noise and fluctuations in the learning process.

        This smoothing provides more consistent and stable targets for the student to learn from,
        increasing training efficacy. Additionally, this decoupling permits more exploration in the
        student without directly affecting the teacher's parameters, preventing the student from
        overfitting to the techer's instantaneous updates.
        """
        # Disable gradient computation
        with torch.no_grad():
            for student_param, teacher_param in zip(
                student_model.parameters(), teacher_model.parameters()
            ):
                teacher_param.data.mul_(other=m).add_(
                    other=student_param.data, alpha=1 - m
                )

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : torch.Tensor
            _description_
        batch_idx : int
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        # Generate random target and context aspect ratio and scale
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        context_scale: float = np.random.uniform(
            self.context_scale[0], self.context_scale[1]
        )

        (
            y_student,  # prediction: (num_target_blocks, batch_size, target_block_size, embed_dim) 
            y_teacher,  # actual values: (num_target_blocks, batch_size, target_block_size, embed_dim)
        ) = self(
            x=batch,  # (batch_size, channels, img_height, img_width)
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("train_loss", loss)

        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Perform a validation step for each image (tensor) in the batch of images (list of tensors).

        Parameters
        ----------
        batch : torch.Tensor
            A tensor representing the batch of data (images).
        batch_idx : int
            Index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The aggregated loss for the batch.
        """
        # Generate random target and context aspect ratio and scale
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        context_scale: float = np.random.uniform(
            self.context_scale[0], self.context_scale[1]
        )

        (
            y_student,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
            y_teacher,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        ) = self(
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("val_loss", loss)

        return loss

    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : torch.Tensor
            _description_
        batch_idx : int
            _description_
        dataloader_idx : int
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        # Generate random target and context aspect ratio
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        self.mode = "test"

        return self(  # Return only student embedding using the student (ViT) encoder
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=1,
        )  # (batch_size, num_patches, embed_dim)

    def on_after_backward(self) -> None:
        self.update_momentum(self.m)
        self.m += (
            self.momentum_limits[1] - self.momentum_limits[0]
        ) / self.trainer.estimated_stepping_batches

    def configure_optimizers(
        self,
    ) -> Dict[str, Union[Callable, Dict[str, Union[str, Callable]]]]:
        optimizer: Callable = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler: Callable = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


