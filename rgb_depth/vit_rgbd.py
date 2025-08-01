from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from x_transformers import Encoder

from utils.types import ensure_tuple

from patch_embed import PatchEmbed2D, PatchEmbed3D


class RGBDVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans_rgb: int = 3,
        in_chans_dep: int = 1,
        embed_dim: int = 64,
        enc_depth: int = 8,
        num_heads: int = 8,
        post_emb_norm: bool = True,
        post_enc_norm: bool = True,
        layer_dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__()
        self.img_size = ensure_tuple(img_size)
        self.patch_size = ensure_tuple(patch_size)

        self.num_frames = num_frames
        self.is_video = num_frames > 1
        self.tubelet_size = tubelet_size

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed_rgb: nn.Module = (
            PatchEmbed2D(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans_rgb,
                embed_dim=embed_dim,
            )
            if not self.is_video
            else PatchEmbed3D(
                img_size=img_size,
                num_frames=num_frames,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans_rgb,
                embed_dim=embed_dim,
            )
        )

        self.patch_embed_dep: nn.Module = (
            PatchEmbed2D(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans_dep,
                embed_dim=embed_dim,
            )
            if not self.is_video
            else PatchEmbed3D(
                img_size=img_size,
                num_frames=num_frames,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans_dep,
                embed_dim=embed_dim,
            )
        )

        # self.num_patches: int = int(
        #     torch.prod(torch.Tensor(self.patch_embed_rgb.patch_shape)).item()
        # )

        self.num_patches = self.patch_embed_rgb.patch_shape[0] * self.patch_embed_rgb.patch_shape[1]

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.post_emb_norm = post_emb_norm
        self.post_emb_norm_vit = (
            nn.LayerNorm(embed_dim) if self.post_emb_norm else nn.Identity()
        )

        self.layer_dropout = layer_dropout

        self.encoder = Encoder(  # student encoder
            dim=embed_dim,
            heads=num_heads,
            depth=enc_depth,
            layer_dropout=self.layer_dropout,
        )

        self.post_enc_norm = post_enc_norm
        self.post_enc_norm_vit = (
            nn.LayerNorm(embed_dim) if self.post_enc_norm else nn.Identity()
        )  # student encoder

    def forward_vit(
        self,
        x_rgb: torch.Tensor,
        x_dep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_embed_only: bool = False,
    ) -> torch.Tensor:
        
        x_rgb = self.patch_embed_rgb(x_rgb)
        x_rgb = x_rgb + self.pos_embedding
        x_rgb = self.post_emb_norm_vit(x_rgb)
        if x_dep is not None:
            x_dep = self.patch_embed_dep(x_dep)
            x_dep = x_dep + self.pos_embedding
            x_dep = self.post_emb_norm_vit(x_dep)
        if patch_embed_only: # Training
            return x_rgb, x_dep
        # Inference
        x_rgb = self.encoder(x_rgb, attn_mask=attention_mask)
        x_rgb = self.post_enc_norm_vit(x_rgb)
        return x_rgb, None
        
        
        


def vit_nano(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=64,
        enc_depth=8,
        num_heads=8,
        **kwargs,
    )


def vit_tiny(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=192,
        enc_depth=12,
        num_heads=8,
        **kwargs,
    )


def vit_small(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=384,
        enc_depth=18,
        num_heads=8,
        **kwargs,
    )


def vit_base(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=768,
        enc_depth=18,
        num_heads=12,
        **kwargs,
    )


def vit_large(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=1024,
        enc_depth=24,
        num_heads=16,
        **kwargs,
    )


def vit_huge(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=1280,
        enc_depth=32,
        num_heads=16,
        **kwargs,
    )