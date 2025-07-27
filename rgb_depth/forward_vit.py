from vit_rgbd import RGBDVisionTransformer
import torch

vit_model = RGBDVisionTransformer(
                                img_size= 64,
                                patch_size= 16,
                                num_frames = 1,
                                tubelet_size = 2,
                                in_chans_rgb = 3,
                                in_chans_dep = 1,
                                embed_dim = 64,
                                enc_depth = 8,
                                num_heads = 8,
                                post_emb_norm = True,
                                post_enc_norm = True,
                                layer_dropout = 0.1,
                                ) 


if __name__ == "__main__":
    rgb = torch.randn(3, 64, 64)
    depth = torch.randn(1, 64, 64)

    x_rgb = rgb.repeat(100, 1, 1, 1)
    x_dep = depth.repeat(100, 1, 1, 1)

    print(vit_model.forward_vit(x_rgb, x_dep))

