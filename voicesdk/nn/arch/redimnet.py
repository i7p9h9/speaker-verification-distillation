import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from voicesdk.nn.feat import features, features_tf
from voicesdk.nn.layers import (
    CausalPaddingNd,
    ConvNeXtLikeBlock,
    LayerNorm,
    ResBasicBlock,
    TransformerEncoderLayer,
    pooling,
    to1d,
    to1d_tfopt,
    to2d,
    to2d_tfopt,
    weigth1d,
)


# ------------------------------------------
#              Main blocks
# ------------------------------------------
class ConvBlock2d(nn.Module):
    def __init__(self, c, f, kernel_sizes=[(3, 3)], block_type="convnext_like", Gdiv=1):
        super().__init__()
        causal = False
        if block_type.endswith("-causal"):
            causal = True
            block_type = block_type.replace("-causal", "")

        conv_next_like_block = functools.partial(
            ConvNeXtLikeBlock, dim=2, kernel_sizes=kernel_sizes, Gdiv=Gdiv, padding="same", causal=causal
        )
        res_basic_block = functools.partial(
            ResBasicBlock,
            inc=c,
            outc=c,
            num_freq=f,
            stride=1,
            se_channels=min(64, max(c, 32)),
            Gdiv=Gdiv,
            causal=causal,
        )
        if block_type == "convnext_like":
            self.conv_block = conv_next_like_block(c, activation="gelu")
        elif block_type == "convnext_like_glu":
            self.conv_block = conv_next_like_block(c, activation="gelu", glu=True)
        elif block_type == "convnext_like_relu":
            self.conv_block = conv_next_like_block(c, activation="relu")
        elif block_type == "basic_resnet":
            self.conv_block = res_basic_block(use_fwSE=False)
        elif block_type == "basic_resnet_fwse":
            self.conv_block = res_basic_block(use_fwSE=True)
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.conv_block(x)


# ------------------------------------------
#                1D block
# ------------------------------------------


class PosEncConv(nn.Module):
    def __init__(self, C, ks, groups=None):
        super().__init__()
        assert ks % 2 == 1
        self.conv = nn.Conv1d(C, C, ks, padding=ks // 2, groups=C if groups is None else groups)
        self.norm = LayerNorm(C, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        return x + self.norm(self.conv(x))


class TimeContextBlock1d(nn.Module):
    def __init__(self, C, hC, pos_ker_sz=59, block_type="att", red_dim_conv=None, exp_dim_conv=None, **kwargs):
        super().__init__()
        assert pos_ker_sz

        causal = False
        if block_type.endswith("-causal"):
            causal = True
            block_type = block_type.replace("-causal", "")

        self.red_dim_conv = nn.Sequential(nn.Conv1d(C, hC, 1), LayerNorm(hC, eps=1e-6, data_format="channels_first"))

        if block_type == "fc":
            self.tcm = nn.Sequential(
                nn.Conv1d(hC, hC * 2, 1),
                LayerNorm(hC * 2, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
                nn.Conv1d(hC * 2, hC, 1),
            )
        elif block_type == "conv":
            # Just large kernel size conv like in convformer
            self.tcm = nn.Sequential(
                *[
                    ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7, 15, 31], Gdiv=1, padding="same", causal=causal)
                    for i in range(4)
                ]
            )

        elif block_type == "conv_relu":
            # Just large kernel size conv like in convformer
            self.tcm = nn.Sequential(
                *[
                    ConvNeXtLikeBlock(
                        hC, dim=1, kernel_sizes=[7, 15, 31], Gdiv=1, padding="same", activation="relu", causal=causal
                    )
                    for i in range(4)
                ]
            )

        elif block_type == "conv_fov_66":
            # Just large kernel size conv like in convformer
            self.tcm = nn.Sequential(
                *[
                    ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[3, 7, 15], Gdiv=1, padding="same", causal=causal)
                    for i in range(3)
                ]
            )

        elif block_type == "convx2":
            # Just large kernel size conv like in convformer
            self.tcm = nn.Sequential(
                *[
                    ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding="same", causal=causal)
                    for i in range(2)
                ]
            )

        elif block_type == "conv_fov_80":
            # Just large kernel size conv like in convformer
            self.tcm = nn.Sequential(
                *[
                    ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7, 21], Gdiv=1, padding="same", causal=causal)
                    for i in range(4)
                ]
            )

        # elif block_type == 'tdnn':
        #     # Small ECAPA-TDNN based block
        #     self.tcm = ECAPA_TDNN(
        #                     in_channels=hC, out_channels=hC,
        #                     scale=8, blocks_setup=[(3,2), (3,3)])

        # elif block_type == 'gru':
        #     # Just GRU
        #     self.tcm = nn.Sequential(
        #         GRU(
        #             input_size=hC, hidden_size=hC,
        #             num_layers=1, bias=True, batch_first=False,
        #             dropout=0.0, bidirectional=True
        #         ), nn.Conv1d(2*hC, hC, 1)
        #     )
        elif block_type == "att":
            # Basic Transformer self-attention encoder block
            pre_layers = []
            if causal:
                pre_layers.append(CausalPaddingNd(dim=1, kernel_size=pos_ker_sz, dilation=1))
            self.tcm = nn.Sequential(
                *pre_layers,
                PosEncConv(hC, ks=pos_ker_sz, groups=hC),
                TransformerEncoderLayer(n_state=hC, n_mlp=hC * 2, n_head=4, causal=causal),
            )
        # elif block_type == 'tdnn+att':
        #     # Basic Transformer self-attention encoder block
        #     self.tcm = nn.Sequential(
        #         ECAPA_TDNN(
        #             in_channels=hC, out_channels=hC,
        #             scale=6, blocks_setup=[(3,2), (3,3)]),
        #         LayerNorm(hC, eps=1e-6, data_format="channels_first"),
        #         PosEncConv(hC, ks=pos_ker_sz, groups=hC),
        #         TransformerEncoderLayer(
        #             n_state=hC,
        #             n_mlp=hC,
        #             n_head=4
        #         )
        #     )
        elif block_type == "conv+att":
            # Basic Transformer self-attention encoder block
            self.tcm = nn.Sequential(
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding="same", causal=causal),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], Gdiv=1, padding="same", causal=causal),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], Gdiv=1, padding="same", causal=causal),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], Gdiv=1, padding="same", causal=causal),
                TransformerEncoderLayer(n_state=hC, n_mlp=hC, n_head=4, causal=causal),
            )
        else:
            raise NotImplementedError()

        self.exp_dim_conv = nn.Conv1d(hC, C, 1)

    def forward(self, x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x


# Custom wrapper to allow overwriting
def custom_partial(func, **fixed_kwargs):
    def wrapper(**kwargs):
        # Merge fixed kwargs with new kwargs
        merged_kwargs = {**fixed_kwargs, **kwargs}
        return func(**merged_kwargs)

    return wrapper


# ------------------------------------------
#                 ReDimNet
# ------------------------------------------
class ReDimNet(nn.Module):
    def __init__(
        self,
        F=72,
        C=12,
        spec_dims=1,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            # stride, num_blocks, conv_exp, kernel_size, layer_ext, att_block_red
            (1, 2, 1, [(3, 3)], None),  # 16
            (2, 3, 1, [(3, 3)], None),  # 32
            (3, 4, 1, [(3, 3)], 8),  # 64, (72*12 // 8) = 108 - channels in attention block
            (2, 5, 1, [(3, 3)], 8),  # 128
            (1, 5, 1, [(7, 1)], 8),  # 128 # TDNN - time context
            (2, 3, 1, [(3, 3)], 8),  # 256
        ],
        group_divisor=1,
        out_channels=512,
        feat_agg_dropout=0.0,
        # -----------------------
        #     Subnet stuff
        # -----------------------
        return_2d_output=False,
        return_all_outputs=False,
        offset_fm_weights=0,
        is_subnet=False,
    ):
        super().__init__()
        self.F = F
        self.C = C

        self.block_1d_type = block_1d_type
        self.block_2d_type = block_2d_type

        self.stages_setup = stages_setup

        self.feat_agg_dropout = feat_agg_dropout
        self.return_2d_output = return_2d_output

        # Subnet stuff
        self.is_subnet = is_subnet
        self.offset_fm_weights = offset_fm_weights
        self.return_all_outputs = return_all_outputs

        self.build(F, C, spec_dims, stages_setup, group_divisor, out_channels, offset_fm_weights, is_subnet)

    def build(self, F, C, spec_dims, stages_setup, group_divisor, out_channels, offset_fm_weights, is_subnet):
        self.F = F
        self.C = C

        c = C
        f = F
        s = 1
        self.num_stages = len(stages_setup)

        if not is_subnet:
            self.stem = nn.Sequential(
                nn.Conv2d(spec_dims, int(c), kernel_size=3, stride=1, padding="same"),
                LayerNorm(int(c), eps=1e-6, data_format="channels_first"),
                to1d(),
            )
        else:
            self.stem = nn.Sequential(
                weigth1d(N=offset_fm_weights, C=F * C),
                # to2d(f=F,c=C),
                # nn.Conv2d(C, C, kernel_size=3, stride=1, padding='same'),
                # LayerNorm(C, eps=1e-6, data_format="channels_first"),
                # to1d()
            )

        Block1d = functools.partial(TimeContextBlock1d, block_type=self.block_1d_type)
        Block2d = functools.partial(ConvBlock2d, block_type=self.block_2d_type)

        self.stages_cfs = []
        for stage_ind, (stride, num_blocks, conv_exp, kernel_sizes, att_block_red) in enumerate(stages_setup):
            assert stride in [1, 2, 3, 4]

            # Pool frequencies & expand channels if needed
            num_feats_to_weight = offset_fm_weights + stage_ind + 1
            layers = [
                weigth1d(
                    N=num_feats_to_weight,
                    C=F * C if num_feats_to_weight > 1 else 1,
                    requires_grad=num_feats_to_weight > 1,
                ),
                to2d(f=f, c=c),
                nn.Conv2d(
                    int(c), int(stride * c * conv_exp), kernel_size=(stride, 1), stride=(stride, 1), padding=0, groups=1
                ),
            ]

            self.stages_cfs.append((c, f))

            c = stride * c
            assert f % stride == 0
            f = f // stride

            for block_ind in range(num_blocks):
                # ConvBlock2d(f, c, block_type="convnext_like", Gdiv=1)
                layers.append(Block2d(c=int(c * conv_exp), f=f, kernel_sizes=kernel_sizes, Gdiv=group_divisor))

            if conv_exp != 1:
                # Squeeze back channels to align with ReDimNet c+f reshaping:
                _group_divisor = group_divisor
                # if c // group_divisor == 0:
                # _group_divisor = c
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            int(c * conv_exp),
                            c,
                            kernel_size=(3, 3),
                            stride=1,
                            padding="same",
                            groups=c // _group_divisor if _group_divisor is not None else 1,
                        ),
                        nn.BatchNorm2d(
                            c,
                            eps=1e-6,
                        ),
                        nn.ReLU() if (("relu" in self.block_1d_type) and ("relu" in self.block_2d_type)) else nn.GELU(),
                        nn.Conv2d(c, c, 1),
                    )
                )

            layers.append(to1d())

            if att_block_red is not None:
                layers.append(Block1d(C * F, hC=(C * F) // att_block_red))

            setattr(self, f"stage{stage_ind}", nn.Sequential(*layers))

        num_feats_to_weight_fin = offset_fm_weights + len(stages_setup) + 1
        self.fin_wght1d = weigth1d(N=num_feats_to_weight_fin, C=F * C, requires_grad=num_feats_to_weight > 1)

        if out_channels is not None:
            self.mfa = nn.Sequential(
                nn.Conv1d(self.F * self.C, out_channels, kernel_size=1, padding="same"),
                # LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
                nn.BatchNorm1d(out_channels, affine=True),
            )
        else:
            self.mfa = nn.Identity()

        if self.return_2d_output:
            self.fin_to2d = to2d(f=f, c=c)
        else:
            self.fin_to2d = nn.Identity()

    def run_stage(self, prev_outs_1d, stage_ind):
        stage = getattr(self, f"stage{stage_ind}")
        x = stage(prev_outs_1d)
        return x

    def forward(self, inp):
        if not self.is_subnet:
            x = self.stem(inp)
            # outputs_1d = [self.to1d(x)]
            outputs_1d = [x]
        else:
            assert isinstance(inp, list)
            outputs_1d = list(inp)
            x = self.stem(inp)
            # outputs_1d.append(self.to1d(x))
            outputs_1d.append(x)

        for stage_ind in range(self.num_stages):
            # outputs_1d.append(self.run_stage(outputs_1d,stage_ind))
            outputs_1d.append(
                F.dropout(self.run_stage(outputs_1d, stage_ind), p=self.feat_agg_dropout, training=self.training)
            )
        # x = self.weigth1d(outputs_1d,-1)
        x = self.fin_wght1d(outputs_1d)
        outputs_1d.append(x)
        x = self.mfa(self.fin_to2d(x))

        if self.return_all_outputs:
            return x, outputs_1d
        else:
            return x


class ReDimNetTFOpt(nn.Module):
    def __init__(
        self,
        F=72,
        C=12,
        spec_dims=1,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            # stride, num_blocks, conv_exp, kernel_size, layer_ext, att_block_red
            (1, 2, 1, [(3, 3)], None),  # 16
            (2, 3, 1, [(3, 3)], None),  # 32
            (3, 4, 1, [(3, 3)], 8),  # 64, (72*12 // 8) = 108 - channels in attention block
            (2, 5, 1, [(3, 3)], 8),  # 128
            (1, 5, 1, [(7, 1)], 8),  # 128 # TDNN - time context
            (2, 3, 1, [(3, 3)], 8),  # 256
        ],
        group_divisor=1,
        out_channels=512,
        feat_agg_dropout=0.0,
        # -----------------------
        #     Subnet stuff
        # -----------------------
        return_2d_output=False,
        return_all_outputs=False,
        offset_fm_weights=0,
        is_subnet=False,
    ):
        super().__init__()
        self.F = F
        self.C = C

        self.block_1d_type = block_1d_type
        self.block_2d_type = block_2d_type

        self.stages_setup = stages_setup

        self.feat_agg_dropout = feat_agg_dropout
        self.return_2d_output = return_2d_output

        # Subnet stuff
        self.is_subnet = is_subnet
        self.offset_fm_weights = offset_fm_weights
        self.return_all_outputs = return_all_outputs

        self.build(F, C, spec_dims, stages_setup, group_divisor, out_channels, offset_fm_weights, is_subnet)

    def build(self, F, C, spec_dims, stages_setup, group_divisor, out_channels, offset_fm_weights, is_subnet):
        self.F = F
        self.C = C

        c = C
        f = F
        s = 1
        self.num_stages = len(stages_setup)

        if not is_subnet:
            self.stem = nn.Sequential(
                nn.Conv2d(spec_dims, int(c), kernel_size=3, stride=1, padding="same"),
                LayerNorm(int(c), eps=1e-6, data_format="channels_first"),
                to1d_tfopt(),
            )
        else:
            self.stem = nn.Sequential(
                weigth1d(N=offset_fm_weights, C=F * C),
                to2d_tfopt(f=F, c=C),
                nn.Conv2d(C, C, kernel_size=3, stride=1, padding="same"),
                LayerNorm(C, eps=1e-6, data_format="channels_first"),
                to1d_tfopt(),
            )

        Block1d = functools.partial(TimeContextBlock1d, block_type=self.block_1d_type)
        Block2d = functools.partial(ConvBlock2d, block_type=self.block_2d_type)

        self.stages_cfs = []
        for stage_ind, (stride, num_blocks, conv_exp, kernel_sizes, att_block_red) in enumerate(stages_setup):
            assert stride in [1, 2, 3, 4]

            # Pool frequencies & expand channels if needed
            num_feats_to_weight = offset_fm_weights + stage_ind + 1
            layers = [
                weigth1d(
                    N=num_feats_to_weight,
                    C=F * C if num_feats_to_weight > 1 else 1,
                    requires_grad=num_feats_to_weight > 1,
                ),
                to2d_tfopt(f=f, c=c),
                nn.Conv2d(
                    int(c), int(stride * c * conv_exp), kernel_size=(1, stride), stride=(1, stride), padding=0, groups=1
                ),
            ]

            self.stages_cfs.append((c, f))

            c = stride * c
            assert f % stride == 0
            f = f // stride

            for block_ind in range(num_blocks):
                # ConvBlock2d(f, c, block_type="convnext_like", Gdiv=1)
                layers.append(Block2d(c=int(c * conv_exp), f=f, Gdiv=group_divisor))

            if conv_exp != 1:
                # Squeeze back channels to align with ReDimNet c+f reshaping:
                _group_divisor = group_divisor
                # if c // group_divisor == 0:
                # _group_divisor = c
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            int(c * conv_exp),
                            c,
                            kernel_size=(3, 3),
                            stride=1,
                            padding="same",
                            groups=c // _group_divisor if _group_divisor is not None else 1,
                        ),
                        nn.BatchNorm2d(
                            c,
                            eps=1e-6,
                        ),
                        nn.ReLU() if (("relu" in self.block_1d_type) and ("relu" in self.block_2d_type)) else nn.GELU(),
                        nn.Conv2d(c, c, 1),
                    )
                )

            layers.append(to1d_tfopt())

            if att_block_red is not None:
                layers.append(Block1d(C * F, hC=(C * F) // att_block_red))

            setattr(self, f"stage{stage_ind}", nn.Sequential(*layers))

        num_feats_to_weight_fin = offset_fm_weights + len(stages_setup) + 1
        self.fin_wght1d = weigth1d(N=num_feats_to_weight_fin, C=F * C, requires_grad=num_feats_to_weight > 1)

        if out_channels is not None:
            self.mfa = nn.Sequential(
                nn.Conv1d(self.F * self.C, out_channels, kernel_size=1, padding="same"),
                # LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
                nn.BatchNorm1d(out_channels, affine=True),
            )
        else:
            self.mfa = nn.Identity()

        if self.return_2d_output:
            self.fin_to2d = to2d_tfopt(f=f, c=c)
        else:
            self.fin_to2d = nn.Identity()

    def run_stage(self, prev_outs_1d, stage_ind):
        stage = getattr(self, f"stage{stage_ind}")
        x = stage(prev_outs_1d)
        return x

    def forward(self, inp):
        if not self.is_subnet:
            x = self.stem(inp)
            # outputs_1d = [self.to1d(x)]
            outputs_1d = [x]
        else:
            assert isinstance(inp, list)
            outputs_1d = list(inp)
            x = self.stem(inp)
            # outputs_1d.append(self.to1d(x))
            outputs_1d.append(x)

        for stage_ind in range(self.num_stages):
            # outputs_1d.append(self.run_stage(outputs_1d,stage_ind))
            outputs_1d.append(
                F.dropout(self.run_stage(outputs_1d, stage_ind), p=self.feat_agg_dropout, training=self.training)
            )
        # x = self.weigth1d(outputs_1d,-1)
        x = self.fin_wght1d(outputs_1d)
        outputs_1d.append(x)
        x = self.mfa(self.fin_to2d(x))

        if self.return_all_outputs:
            return x, outputs_1d
        else:
            return x


class ReDimNetWrap(nn.Module):
    def __init__(
        self,
        F=72,
        C=16,
        spec_dims=1,
        block_1d_type="att",
        block_2d_type="convnext_like",
        # Default setup: M version:
        stages_setup=[
            # stride, num_blocks, kernel_size, layer_ext, drop_path_prob, att_block_red
            (1, 2, 1, [(3, 3)], 12),
            (2, 2, 1, [(3, 3)], 12),
            (1, 3, 1, [(3, 3)], 12),
            (2, 4, 1, [(3, 3)], 8),
            (1, 4, 1, [(3, 3)], 8),
            (2, 4, 1, [(3, 3)], 4),
        ],
        group_divisor=4,
        out_channels=None,
        # -------------------------
        embed_dim=192,
        num_classes=None,
        class_dropout=0.0,
        feat_agg_dropout=0.0,
        head_activation=None,
        hop_length=160,
        pooling_func="ASTP",
        feat_type="pt",
        global_context_att=False,
        emb_bn=False,
        return_2d_output=False,
        # -------------------------
        spec_params=dict(
            do_spec_aug=False,
            freq_mask_width=(0, 6),
            time_mask_width=(0, 8),
        ),
        # -------------------------
        return_all_outputs=False,
        tf_optimized_arch=False,
        use_feats=True,
        # offset_fm_weights = 0,
        # is_subnet = False
    ):
        super().__init__()

        self.return_all_outputs = return_all_outputs
        self.tf_optimized_arch = tf_optimized_arch

        if tf_optimized_arch:
            _ReDimNet = ReDimNetTFOpt
        else:
            _ReDimNet = ReDimNet

        self.backbone = _ReDimNet(
            F,
            C,
            spec_dims,
            block_1d_type,
            block_2d_type,
            stages_setup,
            group_divisor,
            out_channels,
            feat_agg_dropout=feat_agg_dropout,
            return_2d_output=return_2d_output,
            return_all_outputs=return_all_outputs,
            offset_fm_weights=0,
            is_subnet=False,
        )
        print(f"init redimnet with use_feats: {use_feats}")
        if not use_feats:
            self.spec = None
        elif feat_type in ["pt", "pt_mel"]:
            self.spec = features.MelBanks(n_mels=F, hop_length=hop_length, **spec_params)
        elif feat_type in ["tf", "tf_mel"]:
            self.spec = features_tf.TFMelBanks(n_mels=F, hop_length=hop_length, **spec_params)
        elif feat_type == "tf_spec":
            self.spec = features_tf.TFSpectrogram(**spec_params)
        elif feat_type == "pt_stft":
            self.spec = features.STFT(**spec_params)
        elif feat_type is None:
            self.spec = None

        if out_channels is None:
            out_channels = C * F

        if embed_dim is not None:
            self.pool = getattr(pooling, pooling_func)(in_dim=out_channels, global_context_att=global_context_att)

            self.pool_out_dim = self.pool.get_out_dim()
            self.bn = nn.BatchNorm1d(self.pool_out_dim)
            self.linear = nn.Linear(self.pool_out_dim, embed_dim)
            self.embed_dim = embed_dim
            self.emb_bn = emb_bn
            if emb_bn:  # better in SSL for SV
                self.bn2 = nn.BatchNorm1d(embed_dim)
            else:
                self.bn2 = None

            if num_classes is not None:
                self.cls_head = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=class_dropout, inplace=False),
                    nn.Linear(embed_dim, num_classes),
                    eval(head_activation) if head_activation is not None else nn.Identity(),
                )
            else:
                self.cls_head = None
        else:
            self.pool = nn.Identity()
            self.bn = nn.Identity()
            self.linear = nn.Identity()
            self.bn2 = None
            self.cls_head = None

    def forward(self, x):
        if self.spec is not None:
            x = self.spec(x)
        if self.tf_optimized_arch:
            x = x.permute(0, 2, 1)

        if x.ndim == 3:
            x = x.unsqueeze(1)

        if self.return_all_outputs:
            out, all_outs_1d = self.backbone(x)
        else:
            out = self.backbone(x)

        out = self.bn(self.pool(out))
        out = self.linear(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.cls_head is not None:
            out = self.cls_head(out)

        if self.return_all_outputs:
            return out, all_outs_1d
        else:
            return out


class DownReDimNet(nn.Module):
    def __init__(
        self,
        F=72,
        C=16,
        spec_dims=1,
        block_1d_type="att",
        block_2d_type="convnext_like",
        # Default setup: M version:
        stages_setup=[
            # stride, num_blocks, kernel_size, layer_ext, drop_path_prob, att_block_red
            (1, 2, 1, [(3, 3)], 12),
            (2, 2, 1, [(3, 3)], 12),
            (1, 3, 1, [(3, 3)], 12),
            (2, 4, 1, [(3, 3)], 8),
            (1, 4, 1, [(3, 3)], 8),
            (2, 4, 1, [(3, 3)], 4),
        ],
        group_divisor=4,
        out_channels=None,
        # -------------------------
        feat_agg_dropout=0.0,
        hop_length=160,
        feat_type="pt",
        return_2d_output=False,
        # -------------------------
        spec_params=dict(
            do_spec_aug=False,
            freq_mask_width=(0, 6),
            time_mask_width=(0, 8),
        ),
        # -------------------------
    ):
        super().__init__()

        self.backbone = ReDimNet(
            F,
            C,
            spec_dims,
            block_1d_type,
            block_2d_type,
            stages_setup,
            group_divisor,
            out_channels,
            feat_agg_dropout=feat_agg_dropout,
            return_2d_output=return_2d_output,
            return_all_outputs=False,
            offset_fm_weights=0,
            is_subnet=False,
        )
        if feat_type in ["pt", "pt_mel"]:
            self.spec = features.MelBanks(n_mels=F, hop_length=hop_length, **spec_params)
        elif feat_type in ["tf", "tf_mel"]:
            self.spec = features_tf.TFMelBanks(n_mels=F, hop_length=hop_length, **spec_params)
        elif feat_type == "tf_spec":
            self.spec = features_tf.TFSpectrogram(**spec_params)
        elif feat_type == "pt_stft":
            self.spec = features.STFT(**spec_params)

    def forward(self, x):
        x = self.spec(x)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.backbone(x)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def resolve_module_conf(module_config):
    if isinstance(module_config, dict):
        module = eval(module_config["type"])(*module_config.get("args", []), **module_config.get("kwargs", {}))
        trainable = module_config.get("trainable", True)
        if trainable is not None:
            for param in module.parameters():
                param.requires_grad = trainable
        return module
    else:
        raise NotImplementedError()()


class SequentialModel(nn.Module):
    def __init__(self, submodules=[]):
        super(SequentialModel, self).__init__()
        print(f"submodules : {submodules}")
        self.submodules = nn.Sequential(*[resolve_module_conf(sm) for sm in submodules])

    def forward(self, x):
        return self.submodules(x)


class ReDimNetSubNet(nn.Module):
    def __init__(
        self,
        backbone_exp_dir_ep: tuple = None,
        backbone_config: dict = None,
        backbone_weights: str = None,
        freeze_backbone=True,
        add_emb_to_outs=False,
        subnets_configs={
            "downstream_0": dict(
                F=72 // 2,
                C=16 * 2,
                stages_setup=[
                    # stride, num_blocks, kernel_size, layer_ext, drop_path_prob, att_block_red
                    (1, 2, 1, [(3, 3)], 48),
                    (2, 2, 1, [(3, 3)], 48),
                    (1, 2, 1, [(3, 3)], 48),
                ],
                group_divisor=1,
                out_channels=None,
                return_all_outputs=False,
                offset_fm_weights=0,
                is_subnet=False,
            )
        },
    ):
        super().__init__()
        # backbone_config['is_subnet'] = False
        # backbone_config['offset_fm_weights'] = 0
        if backbone_exp_dir_ep is None:
            backbone_config["return_all_outputs"] = True
            backbone = ReDimNetWrap(**backbone_config)
            if freeze_backbone:
                backbone = freeze_model(backbone)
            if backbone_weights is not None:
                state_dict = torch.load(backbone_weights, map_location=torch.device("cpu"))
                if list(state_dict.keys())[0].startswith("backbone.backbone."):
                    state_dict = {k[9:]: v for k, v in state_dict.items()}
                load_res = backbone.load_state_dict(state_dict, strict=False)
                # print(f"{list(backbone.state_dict().keys())[-10:]}")
                print(f"Backbone weights load result : {load_res}")
        else:
            from wespeaker.utils.utils import load_experiment_or_asset_model

            backbone = load_experiment_or_asset_model(*backbone_exp_dir_ep).eval()
            backbone.backbone.return_all_outputs = True
            backbone.return_all_outputs = True
            if freeze_backbone:
                backbone = freeze_model(backbone)

        self.freeze_backbone = freeze_backbone
        self.add_emb_to_outs = add_emb_to_outs
        self.backbone = backbone

        subnets = {}
        subnets_heads = {}
        for sn_name, sn_cfg_all in subnets_configs.items():
            if "subnet_config" in sn_cfg_all:
                sn_cfg = sn_cfg_all["subnet_config"]
            else:
                sn_cfg = sn_cfg_all

            if "prehead_config" in sn_cfg_all:
                prehead_cfg = sn_cfg_all["prehead_config"]
                prehead = resolve_module_conf(prehead_cfg)
            else:
                prehead = None

            if "head_config" in sn_cfg_all:
                head_cfg = sn_cfg_all["head_config"]
                subnets_heads[sn_name] = resolve_module_conf(head_cfg)
            else:
                subnets_heads[sn_name] = nn.Identity()

            sn_cfg["is_subnet"] = True
            sn_cfg["return_all_outputs"] = False
            sn_cfg["offset_fm_weights"] = len(self.backbone.backbone.stages_setup) + 2

            subnet_body = ReDimNet(**sn_cfg)

            if prehead is not None:
                subnets[sn_name] = nn.Sequential(subnet_body, prehead)
            else:
                subnets[sn_name] = subnet_body

        self.subnets = nn.ModuleDict(subnets)
        self.subnets_heads = nn.ModuleDict(subnets_heads)

    def forward(self, x, y=None):
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                bb_out, bb_outs = self.backbone(x)
        else:
            self.backbone.train()
            bb_out, bb_outs = self.backbone(x)

        outs = {}
        if self.add_emb_to_outs:
            outs["embedding"] = bb_out

        for sn_name, sn_mod in self.subnets.items():
            sn_out = sn_mod(bb_outs)
            if y is not None:
                sn_out = self.subnets_heads[sn_name](sn_out, y)
            else:
                sn_out = self.subnets_heads[sn_name](sn_out)
            outs[sn_name] = sn_out
        if len(outs) == 1:
            return outs[sn_name]
        else:
            return outs


def ReDimNetCustom(**kwargs):
    return ReDimNetWrap(**kwargs)


# XS (C=12,F=72,oF=9) model:
#    - 0.25 GMACs
#    - 1.092.692 params ~ 1.1M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 9
def ReDimNet_XSv0_c12f72o9x15ms_nxt_att(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=12,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 2, 1, [(3, 3)], 24),
            (2, 2, 1, [(3, 3)], None),
            (1, 3, 1, [(3, 3)], 24),
            (2, 4, 1, [(3, 3)], None),
            (1, 4, 1, [(3, 3)], 24),
            (2, 4, 1, [(3, 3)], None),
        ],
        group_divisor=1,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# XS (C=16,F=72,oF=9) model:
#    - 0.29 GMACs
#    - 1.135.696 params ~ 1.1M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 9
def ReDimNet_XSv0_c16f72o9x15ms_nxt_att(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=16,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 2, 1, [(3, 3)], 32),
            (2, 2, 1, [(3, 3)], None),
            (1, 3, 1, [(3, 3)], 32),
            (2, 3, 1, [(3, 3)], None),
            (1, 3, 1, [(3, 3)], 32),
            (2, 2, 1, [(3, 3)], None),
        ],
        group_divisor=1,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=False,
        emb_bn=False,
    )


# S (C=14,F=72,oF=9) model:
#    - 0.39 GMACs
#    - 2.051.846 params ~ 2.0M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 9
def ReDimNet_Sv0_c14f72o9x15ms_nxt_att(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=14,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 2, 1, [(3, 3)], 21),
            (2, 2, 1, [(3, 3)], 21),
            (1, 3, 1, [(3, 3)], 14),
            (2, 3, 1, [(3, 3)], 14),
            (1, 3, 1, [(3, 3)], 14),
            (2, 3, 1, [(3, 3)], 12),
        ],
        group_divisor=1,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# S (C=16,F=72,oF=9) model:
#    - 0.62 GMACs
#    - 2.193.846 params ~ 2.2M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 9
def ReDimNet_Sv0_c16f72o9x15ms_rn_catt(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=16,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 2, 1, [(3, 3)], 32),
            (2, 2, 1, [(3, 3)], 32),
            (1, 3, 1, [(3, 3)], 24),
            (2, 3, 1, [(3, 3)], 24),
            (1, 3, 1, [(3, 3)], 16),
            (2, 3, 1, [(3, 3)], 16),
        ],
        group_divisor=1,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# M (C=16,F=72,oF=3) model:
#    - 1.83 GMACs
#    - 3.311.784 params ~ 3.3M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 9
def ReDimNet_SMv1_c16f72o9x15ms_rn_catt(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=16,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 6, 2, [(3, 3)], 32),
            (2, 6, 2, [(3, 3)], 32),
            (1, 8, 2, [(3, 3)], 16),
            (2, 10, 1, [(3, 3)], 16),
            (1, 10, 1, [(3, 3)], 8),
            (2, 3, 1, [(3, 3)], 8),
        ],
        group_divisor=1,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


def ReDimNet_SMv2_c16f72o9x15ms_rn_catt(
    embed_dim=192, feat_dim=72, feat_type="pt", use_specaug=False, pooling_func="ASTP", emb_bn=False
):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    spec_params = dict(
        do_spec_aug=True,
        freq_start_bin=24,
        freq_mask_width=(0, 6),
        time_mask_width=(0, 6),
    )
    return ReDimNetWrap(
        F=72,
        C=16,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 4, 2, [(3, 3)], 32),
            (2, 5, 2, [(3, 3)], 32),
            (1, 6, 2, [(3, 3)], 16),
            (2, 8, 1, [(3, 3)], 16),
            (1, 8, 1, [(3, 3)], 8),
            (2, 3, 1, [(3, 3)], 8),
        ],
        group_divisor=16,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        feat_type=feat_type,
        spec_params=spec_params if use_specaug else {},
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# M (C=20,F=72,oF=3) model:
# MACs : 2.266744032
# params : 3295568.0
def ReDimNet_SMv3_c20f72o3x15ms_rn_csfov_v0(embed_dim=192, group_divisor=4, hop_mult=1.8, global_context_att=False):
    return ReDimNetWrap(
        F=72,
        C=20,
        block_1d_type="conv_fov_66",
        block_2d_type="basic_resnet",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 4, 2, [(3, 3)], 20),
            (2, 5, 2, [(3, 3)], 20),
            (1, 6, 2, [(3, 3)], 20),
            (2, 8, 1, [(3, 3)], 20),
            (1, 8, 1, [(3, 3)], 20),
            (2, 3, 1, [(3, 3)], 20),
        ],
        group_divisor=group_divisor,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=embed_dim,
        hop_length=int(160 * hop_mult),
        feat_type="tf",
        spec_params=dict(
            norm_signal=True,
            do_preemph=True,
            do_spec_aug=True,
            freq_start_bin=24,
            freq_mask_width=(0, 6),
            time_mask_width=(0, 6),
        ),
        pooling_func="ASTP",
        # pooling_func='TSTP',
        global_context_att=global_context_att,
        emb_bn=False,
    )


# M (C=24,F=72,oF=3) model:
# MACs : 3.03669072
# params : 3301824.0
def ReDimNet_Mv1_c24f72o3x15ms_rn_csfov_v0(embed_dim=192):
    return ReDimNetWrap(
        F=72,
        C=24,
        block_1d_type="conv_fov_66",
        block_2d_type="basic_resnet",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 4, 2, [(3, 3)], 48),
            (2, 5, 2, [(3, 3)], 48),
            (1, 6, 2, [(3, 3)], 36),
            (2, 8, 1, [(3, 3)], 36),
            (1, 8, 1, [(3, 3)], 24),
            (2, 3, 1, [(3, 3)], 12),
        ],
        group_divisor=2,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        feat_type="tf",
        spec_params=dict(
            norm_signal=True,
            do_preemph=False,
            do_spec_aug=True,
            freq_start_bin=24,
            freq_mask_width=(0, 6),
            time_mask_width=(0, 6),
        ),
        # pooling_func='ASTP',
        pooling_func="TSTP",
        global_context_att=True,
        emb_bn=False,
    )


# M (C=20,F=72,oF=3) model:
#    - 3.32 GMACs
#    - 4.274.264 params ~ 4.3M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 3
def ReDimNet_Mv1_c20f72o3x15ms_rn_catt(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=20,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 6, 4, [(3, 3)], 30),
            (2, 6, 2, [(3, 3)], 30),
            (1, 8, 2, [(3, 3)], 20),
            (2, 10, 1, [(3, 3)], 20),
            (1, 10, 1, [(3, 3)], 10),
            (2, 3, 1, [(3, 3)], 10),
        ],
        group_divisor=1,
        out_channels=None,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# M (C=16,F=72,oF=3) model:
#    - 1.29 GMACs
#    - 4.575.024 params ~ 4.6M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 3 // 2 // 2 = 3
def ReDimNet_Mv0_c16f72o3x15ms_nxt_att(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=16,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 2, 1, [(3, 3)], 16),  # 16
            (2, 3, 1, [(3, 3)], 16),  # 32
            (3, 4, 1, [(3, 3)], 8),  # 64
            (2, 5, 1, [(3, 3)], 4),  # 128
            (2, 3, 1, [(3, 3)], None),  # 256
        ],
        group_divisor=16,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# M (C=16,F=72,oF=9) model:
#    - 0.9 GMACs
#    - 4.737.584 params ~ 4.7M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 9
def ReDimNet_Mv0_c16f72o9x15ms_nxt_att(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=16,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 2, 1, [(3, 3)], 12),
            (2, 2, 1, [(3, 3)], 12),
            (1, 3, 1, [(3, 3)], 12),
            (2, 4, 1, [(3, 3)], 8),
            (1, 4, 1, [(3, 3)], 6),
            (2, 4, 1, [(3, 3)], 4),
        ],
        group_divisor=4,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# M (C=24,F=72,oF=9) model:
#    - 1.22 GMACs
#    - 4.843.880 params ~ 4.8M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 9
def ReDimNet_Mv0_c24f72o9x15ms_nxt_att(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=24,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 2, 1, [(3, 3)], 24),
            (2, 2, 1, [(3, 3)], 24),
            (1, 3, 1, [(3, 3)], 24),
            (2, 4, 1, [(3, 3)], 12),
            (1, 4, 1, [(3, 3)], 12),
            (2, 4, 1, [(3, 3)], 12),
        ],
        group_divisor=4,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# M (C=32,F=72,oF=9) model:
#    - 1.64 GMACs
#    - 4.865.248 params ~ 4.9M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 9
def ReDimNet_Mv0_c32f72o9x15ms_nxt_att(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=32,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            # stride, num_blocks, block_expansion, kernel_sizes, att_block_red
            (1, 2, 1, [(3, 3)], 48),
            (2, 2, 1, [(3, 3)], 48),
            (1, 3, 1, [(3, 3)], 48),
            (2, 4, 1, [(3, 3)], 32),
            (1, 4, 1, [(3, 3)], 24),
            (2, 4, 1, [(3, 3)], 24),
        ],
        group_divisor=4,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# L (C=32,F=72,oF=9) model:
#    - 8.92 GMACs
#    - 7.596.576 params ~ 7.6M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 9
def ReDimNet_Lv0_c32f72o9x15ms_rnfwse_catt(
    embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False
):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=32,
        # block_1d_type = "att",
        block_1d_type="conv+att",
        # block_2d_type = "convnext_like",
        block_2d_type="basic_resnet_fwse",
        # Default setup: M version:
        stages_setup=[
            # stride, num_blocks, kernel_size, layer_ext, drop_path_prob, att_block_red
            (1, 4, 2, [(3, 3)], 48),
            (2, 4, 2, [(3, 3)], 48),
            (1, 6, 2, [(3, 3)], 48),
            (2, 6, 1, [(3, 3)], 32),
            (1, 8, 1, [(3, 3)], 24),
            (2, 4, 1, [(3, 3)], 16),  # 72 * 32 // 16 ->
        ],
        group_divisor=16,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# L (C=32,F=72,oF=9) model:
#    - 8.92 GMACs
#    - 7.596.576 params ~ 7.6M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 2 // 2 = 9
def ReDimNet_Lv0_c32f72o9x15ms_rnfwse_att(
    embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False
):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=32,
        block_1d_type="att",
        block_2d_type="basic_resnet_fwse",
        stages_setup=[
            # stride, num_blocks, kernel_size, layer_ext, drop_path_prob, att_block_red
            (1, 4, 2, [(3, 3)], 48),
            (2, 4, 2, [(3, 3)], 48),
            (1, 6, 2, [(3, 3)], 48),
            (2, 6, 1, [(3, 3)], 32),
            (1, 8, 1, [(3, 3)], 24),
            (2, 4, 1, [(3, 3)], 16),
        ],
        group_divisor=16,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# MACs : 8.857890912
# params : 7596576.0
def ReDimNet_B6_c32f72o9x15ms_rnfwse_catt(
    embed_dim=192, feat_dim=72, feat_type="tf", use_specaug=True, pooling_func="ASTP", emb_bn=False
):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    spec_params = dict(
        do_spec_aug=True,
        freq_start_bin=20,
        freq_mask_width=(0, 6),
        time_mask_width=(0, 6),
    )
    return ReDimNetWrap(
        F=72,
        C=32,
        # block_1d_type = "att",
        block_1d_type="conv+att",
        # block_2d_type = "convnext_like",
        block_2d_type="basic_resnet",
        # Default setup: M version:
        stages_setup=[
            # stride, num_blocks, kernel_size, layer_ext, drop_path_prob, att_block_red
            (1, 4, 2, [(3, 3)], 48),
            (2, 4, 2, [(3, 3)], 48),
            (1, 6, 2, [(3, 3)], 48),
            (2, 6, 1, [(3, 3)], 32),
            (1, 8, 1, [(3, 3)], 24),
            (2, 4, 1, [(3, 3)], 16),  # 72 * 32 // 16 ->
        ],
        group_divisor=16,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        feat_type=feat_type,
        spec_params=spec_params if use_specaug else {},
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )


# L (C=32,F=72,oF=6) model:
#    - 9.87 GMACs
#    - 9.239.672 params ~ 9.2M
# Final prepooling num frequencies:
# 72 MelBanks -> 72 // 2 // 3 // 2 = 6
def ReDimNet_Lv0_c32f72o6x15ms_rn_catt(embed_dim=192, feat_dim=72, feat_type="pt", pooling_func="ASTP", emb_bn=False):
    assert feat_dim == feat_dim  # Dummy string only for back-compatibility with wespeaker
    return ReDimNetWrap(
        F=72,
        C=32,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet",
        stages_setup=[
            # stride, num_blocks, kernel_size, layer_ext, drop_path_prob, att_block_red
            (1, 4, 2, [(3, 3)], 48),
            (2, 4, 2, [(3, 3)], 48),
            (1, 6, 2, [(3, 3)], 48),
            (3, 6, 1, [(3, 3)], 32),
            (1, 8, 1, [(3, 3)], 24),
            (2, 4, 1, [(3, 3)], 16),
        ],
        # group_divisor = 1,
        group_divisor=16,
        out_channels=None,
        # turn_off_1d_bus=False,
        # -------------------------
        embed_dim=192,
        hop_length=int(160 * 1.5),
        pooling_func="ASTP",
        global_context_att=True,
        emb_bn=False,
    )
