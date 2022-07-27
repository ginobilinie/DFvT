import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def convdw(in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )


class DANE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DANE, self).__init__()
        self.channel = channel
        self.fc_spatial = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, 1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_channel = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.LayerNorm(channel//reduction),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_tf,x_cnn):
        #B L C
        x_spatial_mask = self.fc_spatial(x_cnn) # B L 1
        x_channel_mask = self.fc_channel(self.avg_pool(x_tf.permute(0,2,1)).permute(0,2,1)) # B 1 C
        x_mask = self.sigmoid(x_spatial_mask.expand_as(x_cnn) + x_channel_mask.expand_as(x_tf))
        return x_cnn * x_mask + x_tf * (1 - x_mask)




def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops




class TransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.select = DANE(channel = self.dim)
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, x_cnn):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)


        x = self.select(x, x_cnn)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        #select
        flops += self.dim * H * W +  self.dim * (self.dim // 16) * 2 * H * W
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += H  * W * self.dim * 2 * self.dim

        return flops



class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_cnn):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_cnn)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj1 = nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2,padding=1)
        self.proj2 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2,padding=1)
        self.silu = nn.SiLU(inplace=True)
        if norm_layer is not None:
            self.norm = nn.BatchNorm2d(embed_dim)
            self.norm1 = nn.BatchNorm2d(embed_dim//2)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.silu(x)
        x = self.proj2(x)
        x = self.norm(x)
        x = self.silu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

    def flops(self):
        flops=0
        Ho, Wo = self.patches_resolution
        #proj1
        flops1 = Ho/2 * Wo/2 * self.embed_dim/2 * self.in_chans * 9
        #proj2
        flops2 = Ho/4* Wo/4 * self.embed_dim * self.embed_dim/2 * 9
        flops=flops+flops1+flops2
        #norm
        flops += Ho/4 * Wo/4 * self.embed_dim
        flops += Ho/2 * Wo/2 * self.embed_dim/2
        return flops


class ConvBlock(nn.Module):

    def __init__(self,
                 input_resolution,
                 inplanes,
                 stride,
                 groups=1,
                 norm_layer=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        self.input_resolution = input_resolution
        self.inplanes=inplanes
        self.stride = stride
        self.conv1x1_1 = nn.Sequential(
                nn.Conv2d(inplanes,
                         inplanes,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         groups=groups,
                         bias=False),
                norm_layer(inplanes),
                nn.SiLU(inplace=True),
                nn.Conv2d(inplanes,
                         inplanes,
                         kernel_size=3,
                         stride=1,
                         padding=1,
                         groups=inplanes,
                         bias=False),
                norm_layer(inplanes),
                nn.SiLU(inplace=True),
                nn.Conv2d(inplanes,
                         inplanes,
                         kernel_size=3,
                         stride=1,
                         padding=1,
                         groups=inplanes,
                         bias=False),
                norm_layer(inplanes),
                nn.SiLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
                nn.Conv2d(inplanes,
                         inplanes,
                         kernel_size=3,
                         stride=stride,
                         padding=1,
                         groups=inplanes,
                         bias=False),
                norm_layer(inplanes),
                nn.SiLU(inplace=True)


        )
        self.conv1x1_2 = nn.Sequential(
                nn.Conv2d(inplanes,
                         inplanes,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         groups=groups,
                         bias=False),
                norm_layer(inplanes),
                nn.SiLU(inplace=True)
        )



    def forward(self, x):
        out = self.conv1x1_1(x)
        x_out = out
        out = self.conv1(out)

        out = self.conv1x1_2(out)

        out = out.flatten(2).transpose(1, 2)  # B Ph*Pw C

        return x_out,out

    def flops(self):

        flops=0
        Ho, Wo = self.input_resolution
        #con1vx1
        flops += Ho * Wo * self.inplanes * self.inplanes + Ho * Wo * self.inplanes

        #3 * depthwise conv 3x3
        flops += 2 * Ho * Wo * self.inplanes * 9 + 2 * Ho * Wo * self.inplanes
        if self.stride == 2:
            flops +=  Ho/2 * Wo/2 * self.inplanes * 9 + Ho/2 * Wo/2 * self.inplanes
        else:
            flops +=  Ho * Wo * self.inplanes * 9 + Ho * Wo * self.inplanes

        #conv1x1_2
        if self.stride == 2:
            flops += Ho/2 * Wo/2 * self.inplanes * self.inplanes + Ho/2 * Wo/2 * self.inplanes
        else:
            flops += Ho * Wo * self.inplanes * self.inplanes + Ho * Wo * self.inplanes
        #seperable conv
        if self.stride == 2:
            flops += Ho/2 * Wo/2 * self.inplanes * 4 * 9 + Ho/2 * Wo/2 * 4 * self.inplanes * self.inplanes + 5 * Ho/2 * Wo/2 * self.inplanes


        return flops


class DFvT(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** (i_layer+1)),
                                                 patches_resolution[1] // (2 ** (i_layer+1))) if(i_layer<self.num_layers - 1) else (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.convlayers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ConvBlock(inplanes = embed_dim * 2 ** i_layer,
            stride = 2 if i_layer < self.num_layers - 1 else 1,
            input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)))
            self.convlayers.append(layer)

        self.multiresolution_conv = nn.ModuleList()
        for i_layer in range(self.num_layers-1):
            layer = convdw(4 * embed_dim * 2 ** i_layer, embed_dim * 2 ** i_layer, 1)
            self.multiresolution_conv.append(layer)


        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        i=0
        for convlayer,layer in zip(self.convlayers,self.layers):

            B, L, C = x.shape
            x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

            x_dw, x_inter = convlayer(x)
            if i < self.num_layers - 1:
                x_downsample = self.maxpool(x)

                x0 = x_dw[:, :, 0::2, 0::2]  # B C H/2 W/2
                x1 = x_dw[:, :, 1::2, 0::2]  # B C H/2 W/2
                x2 = x_dw[:, :, 0::2, 1::2]  # B C H/2 W/2
                x3 = x_dw[:, :, 1::2, 1::2]  # B C H/2 W/2
                x_dw = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2
                x_dw = x_dw.view(B, 4 * C, int(math.sqrt(L))//2, int(math.sqrt(L))//2)
                x_dw = self.multiresolution_conv[i](x_dw)
                x_dw = x_dw.flatten(2).transpose(1, 2)
                x_downsample = x_downsample.flatten(2).transpose(1, 2)

                x = layer(x_downsample + x_inter,x_dw)
            else:
                x_downsample = x
                x_downsample = x_downsample.flatten(2).transpose(1, 2)

                x = layer(x_downsample + x_inter,x_downsample)

            i = i + 1

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        for i, layer in enumerate(self.convlayers):
            flops += layer.flops()

        #norm
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)

        #head
        flops += self.num_features * self.num_classes
        return flops

    def get_1x_lr_params_NOscale(self):

        b = []

        b.append(self.patch_embed)
        b.append(self.layers)
        b.append(self.head)
        b.append(self.norm)
        params = []
        params_nodecay = []
        for i in range(len(b)):
            for name, param in b[i].named_parameters():
                if len(param.shape) == 1 or name.endswith(".bias"):
                    params_nodecay.append(param)
                else:
                    params.append(param)

        return params, params_nodecay

    def get_10x_lr_params(self):

        b = []
        b.append(self.convlayers)
        b.append(self.multiresolution_conv)
        params = []
        params_nodecay = []
        for i in range(len(b)):
            for name, param in b[i].named_parameters():
                if len(param.shape) == 1 or name.endswith(".bias"):
                    params_nodecay.append(param)
                else:
                    params.append(param)

        return params, params_nodecay
