import torch
from model_rgbd import IDJEPA

rgb =torch.randn(3, 320, 320)
depth = torch.randn(1, 320, 320)

x_rgb = rgb.repeat(100, 1, 1, 1)
x_dep = depth.repeat(100, 1, 1, 1)

experiment_config = {
    "LR": 1e-3,
    "WEIGHT_DECAY": 0.05,
    "TARGET_ASPECT_RATIO": (0.75, 1.5),
    "TARGET_SCALE_INTERVAL": (0.15, 0.2),
    "CONTEXT_ASPECT_RATIO": 1.0,
    "CONTEXT_SCALE": (0.85, 1.0),
    "NUM_TARGET_BLOCKS": 4,
    "M": 0.996,
    "MOMENTUM_LIMITS": (0.996, 1.0),
}

model_config = {
    "IMAGE_SIZE": 320, # <<<<<<<<<<<<< 
    "PATCH_SIZE": 16,
    "IN_CHANS": 3,  
    "POST_EMBED_NORM": True,
    "POST_ENCODE_NORM": True,
    "LAYER_DROPOUT": 0.1,
}

# Basic ViT settings
embed_dim = 64
enc_depth = 4
num_heads = 4
num_layers_decoder = 2

# Instantiate model
model = IDJEPA(
    decoder_depth=num_layers_decoder,
    lr=experiment_config["LR"],
    weight_decay=experiment_config["WEIGHT_DECAY"],
    target_aspect_ratio=experiment_config["TARGET_ASPECT_RATIO"],
    target_scale_interval=experiment_config["TARGET_SCALE_INTERVAL"],
    context_aspect_ratio=experiment_config["CONTEXT_ASPECT_RATIO"],
    context_scale=experiment_config["CONTEXT_SCALE"],
    num_target_blocks=experiment_config["NUM_TARGET_BLOCKS"],
    m=experiment_config["M"],
    momentum_limits=experiment_config["MOMENTUM_LIMITS"],
    img_size=model_config["IMAGE_SIZE"],
    patch_size=model_config["PATCH_SIZE"],
    in_chans_rgb=3,
    in_chans_dep=1,
    embed_dim=embed_dim,
    enc_depth=enc_depth,
    num_heads=num_heads,
    post_emb_norm=model_config["POST_EMBED_NORM"],
    post_enc_norm=model_config["POST_ENCODE_NORM"],
    layer_dropout=model_config["LAYER_DROPOUT"],
    testing_purposes_only=True,  # optional: avoid checkpoint saving
)

if __name__ == "__main__":

#     y_pred, y_true = model(
#         x_rgb=x_rgb,
#         x_dep=x_dep,
#         target_aspect_ratio=1.0,
#         target_scale=0.2,
#         context_aspect_ratio=1.0,
#         context_scale=0.85,
#     )

# print("Prediction shape:", y_pred.shape)
# print("Ground truth shape:", y_true.shape)

    print(model)