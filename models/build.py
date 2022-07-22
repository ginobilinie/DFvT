# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------


from .DFvT import DFvT



def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'DFvT':
        model = DFvT(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.DFvT.PATCH_SIZE,
                                in_chans=config.MODEL.DFvT.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.DFvT.EMBED_DIM,
                                depths=config.MODEL.DFvT.DEPTHS,
                                num_heads=config.MODEL.DFvT.NUM_HEADS,
                                window_size=config.MODEL.DFvT.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.DFvT.MLP_RATIO,
                                qkv_bias=config.MODEL.DFvT.QKV_BIAS,
                                qk_scale=config.MODEL.DFvT.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.DFvT.APE,
                                patch_norm=config.MODEL.DFvT.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                model_size = config.MODEL.DFvT.SIZE)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

