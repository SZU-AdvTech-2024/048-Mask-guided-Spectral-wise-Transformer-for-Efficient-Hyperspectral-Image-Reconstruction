import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    对给定的张量进行截断正态分布初始化，无梯度计算。

    参数:
    tensor (torch.Tensor): 需要初始化的张量。
    mean (float): 正态分布的均值。
    std (float): 正态分布的标准差。
    a (float): 截断区间的下限。
    b (float): 截断区间的上限。

    返回:
    torch.Tensor: 初始化后的张量。
    """
    # 定义正态分布的累积分布函数（CDF）计算函数
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # 检查均值是否超出合理截断范围，若超出则发出警告
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # 以下是进行截断正态分布初始化的主要操作，不进行梯度计算
    with torch.no_grad():
        # 计算截断区间下限对应的CDF值
        l = norm_cdf((a - mean) / std)
        # 计算截断区间上限对应的CDF值
        u = norm_cdf((b - mean) / std)

        # 对张量进行均匀分布采样
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # 对张量进行反误差函数操作
        tensor.erfinv_()

        # 根据标准差和相关系数调整张量值
        tensor.mul_(std * math.sqrt(2.))

        # 添加均值到张量
        tensor.add_(mean)

        # 将张量的值限制在截断区间[a, b]内
        tensor.clamp_(min=a, max=b)

    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    """
    根据指定的参数对输入张量进行方差缩放初始化操作。

    参数:
    tensor (torch.Tensor): 需要进行初始化的张量。
    scale (float, 可选): 缩放因子，默认为1.0。
    mode (str, 可选): 计算分母的模式，可选值为'fan_in'、'fan_out'、'fan_avg'，分别表示使用输入神经元数量、输出神经元数量、输入和输出神经元数量的平均值作为分母，默认为'fan_in'。
    distribution (str, 可选): 初始化所采用的分布类型，可选值为'truncated_normal'、'normal'、'uniform'，默认为'normal'。

    返回:
    None，该函数直接对输入的张量进行原地修改。
    """
    # 计算输入和输出神经元的数量（这里假设 _calculate_fan_in_and_fan_out 函数已正确定义）
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    # 根据指定的模式选择分母
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"Invalid mode {mode}")
    # 计算方差
    variance = scale / denom

    # 根据指定的分布类型对张量进行相应的初始化操作
    if distribution == "truncated_normal":
        std = math.sqrt(variance) /.87962566103423978
        trunc_normal_(tensor, std=std)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def _apply_layer_norm(self, x):
        """
        私有方法，用于对输入张量执行层归一化操作。
        参数：
        x (torch.Tensor): 输入的张量数据。
        返回：
        torch.Tensor: 归一化后的张量数据。
        """
        return self.norm(x)

    def forward(self, x, *args, **kwargs):
        x = self._apply_layer_norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs, step=2):
    """
    对输入的张量进行特定的维度调整操作（类似移位还原操作）。
    参数:
    inputs (torch.Tensor): 输入张量，形状为[bs, 28, 256, 310]，其中bs表示批量大小，
                           本函数将对最后两维进行处理，将其从[256, 310]的维度调整为[256, 256]。
    step (int, 可选): 用于控制移位的步长参数，默认值为2。
    返回:
    torch.Tensor: 经过维度调整后的张量，形状为[bs, 28, 256, 256]。
    """
    # 获取输入张量的各维度大小
    bs, nC, row, col = inputs.shape()
    # 计算下采样比例，这里根据目标输出维度256与当前行维度row的关系来确定
    down_sample = 256 // row
    # 根据下采样比例等因素调整步长
    step = float(step) / float(down_sample * down_sample)
    # 设定输出的列维度大小，初始化为当前行维度大小，后续会据此截取数据
    out_col = row

    # 对每个通道的数据进行移位还原操作
    for i in range(nC):
        start_index = int(step * i)
        end_index = int(step * i) + out_col
        inputs[:, i, :, :out_col] = inputs[:, i, :, start_index:end_index]

    # 返回处理后的张量，截取到设定的输出维度大小
    return inputs[:, :, :, :out_col]


import torch
import torch.nn as nn


class MaskGuidedMechanism(nn.Module):
    """
    MaskGuidedMechanism类，主要用于基于掩码信息进行特征引导处理的机制。
    通过一系列卷积操作和注意力机制相关计算，对输入的掩码特征进行处理，并返回处理后的掩码嵌入表示。
    参数：
    n_feat (int): 输入特征的通道数量，用于定义卷积层的输入和输出通道数。
    """

    def __init__(self, n_feat):
        """
        初始化MaskGuidedMechanism类的实例。
        参数：
        n_feat (int): 特征通道数量，用于初始化卷积层等相关组件。
        """
        super(MaskGuidedMechanism, self).__init__()

        # 定义第一个1x1卷积层，用于对输入特征进行初步变换，保持输入输出通道数一致，有偏置
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)

        # 定义第二个1x1卷积层，同样保持输入输出通道数一致，有偏置，后续将用于配合深度卷积生成注意力图
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)

        # 定义深度可分离卷积层，用于生成注意力图，groups=n_feat表示分组卷积，每组通道单独卷积，实现深度方向的特征提取
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift):
        """
        前向传播函数，执行基于掩码的特征引导处理流程。
        参数：
        mask_shift (torch.Tensor): 输入的掩码特征张量，形状为(batch_size, n_feat, height, width)，即(b, c, h, w)格式。
        返回：
        torch.Tensor: 处理后的掩码嵌入表示，经过一系列卷积、注意力机制计算以及其他相关操作后的结果。
        """
        # 获取输入张量的形状信息，分别对应batch_size、通道数、高度和宽度
        bs, nC, row, col = mask_shift.shape

        # 首先通过第一个1x1卷积层对输入的掩码特征进行变换
        mask_shift = self.conv1(mask_shift)

        # 通过第二个1x1卷积层进一步变换，然后经过深度可分离卷积层，最后通过sigmoid函数生成注意力图
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))

        # 将原始的掩码特征与注意力图对应元素相乘，得到经过注意力加权后的特征表示
        res = mask_shift * attn_map

        # 将加权后的特征与原始掩码特征相加，进行特征融合
        mask_shift = res + mask_shift

        # 调用外部定义的shift_back函数（假设已正确定义）对融合后的掩码特征进行进一步处理，得到最终的掩码嵌入
        mask_emb = shift_back(mask_shift)

        return mask_emb


class MS_MSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        """
        初始化MS_MSA类的实例。
        参数：
        dim (int): 同类定义中的dim参数，传入输入特征维度大小。
        dim_head (int): 同类定义中的dim_head参数，传入每个头的特征维度大小。
        heads (int): 同类定义中的heads参数，传入头的数量。
        """
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        # 定义将输入特征映射为查询（Query）的线性层，无偏置，输出维度为dim_head * heads
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        # 定义将输入特征映射为键（Key）的线性层，无偏置，输出维度为dim_head * heads
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        # 定义将输入特征映射为值（Value）的线性层，无偏置，输出维度为dim_head * heads
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)

        # 定义可学习的缩放参数，用于调整注意力权重，形状为(heads, 1, 1)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))

        # 定义将多头自注意力机制输出映射回原始特征维度的线性投影层，有偏置
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        # 定义位置嵌入模块，由一系列卷积层和激活函数组成，用于为输入特征添加位置信息
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

        # 实例化掩码引导机制模块，传入特征维度dim，用于结合掩码信息处理特征
        self.mm = MaskGuidedMechanism(dim)
        self.dim = dim

    def forward(self, x_in, mask=None):
        """
        前向传播函数，执行多头自注意力计算以及结合位置嵌入和掩码引导的特征处理操作。
        参数：
        x_in (torch.Tensor): 输入的特征张量，形状为[b, h, w, c]，其中b表示批量大小，h、w表示特征图的高度和宽度，c表示特征维度。
        mask (torch.Tensor, 可选): 掩码张量，形状为[1, h, w, c]，用于引导特征处理过程，若未传入则默认为None。
        返回：
        torch.Tensor: 经过多头自注意力计算、位置嵌入添加以及与掩码信息融合后的输出特征张量，形状为[b, h, w, c]。
        """
        b, h, w, c = x_in.shape

        # 将输入特征张量的维度进行调整，方便后续进行线性变换操作，变为[b, h*w, c]
        x = x_in.reshape(b, h * w, c)

        # 通过线性层将输入特征映射为查询（Query）表示
        q_inp = self.to_q(x)
        # 通过线性层将输入特征映射为键（Key）表示
        k_inp = self.to_k(x)
        # 通过线性层将输入特征映射为值（Value）表示
        v_inp = self.to_v(x)

        # 如果传入了掩码信息，利用掩码引导机制模块对掩码进行处理，并调整维度顺序
        if mask is not None:
            mask_attn = self.mm(mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            if b!= 0:
                # 将掩码信息扩展到与输入特征的批量大小一致，方便后续进行元素级操作
                mask_attn = (mask_attn[0, :, :, :]).expand([b, h, w, c])

        # 对查询、键、值以及掩码（如果有）进行维度重排，以便进行多头处理，按照多头的维度进行拆分
        q, k, v, mask_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp, mask_attn.flatten(1, 2) if mask is not None else mask_attn))

        # 如果有掩码信息，将值（Value）与掩码进行元素级乘法，实现掩码引导的特征过滤
        if mask_attn is not None:
            v = v * mask_attn

        # 对查询进行维度转置，方便后续进行矩阵乘法计算注意力权重，变为[b, heads, c, hw]
        q = q.transpose(-2, -1)
        # 对键进行维度转置，变为[b, heads, c, hw]
        k = k.transpose(-2, -1)
        # 对值进行维度转置，变为[b, heads, c, hw]
        v = v.transpose(-2, -1)

        # 对查询进行归一化操作，按照最后一个维度（特征维度）进行L2归一化
        q = F.normalize(q, dim=-1, p=2)
        # 对键进行归一化操作，按照最后一个维度进行L2归一化
        k = F.normalize(k, dim=-1, p=2)

        # 计算注意力权重，通过矩阵乘法得到注意力矩阵，形状为[b, heads, hw, hw]
        attn = (k @ q.transpose(-2, -1))
        # 应用可学习的缩放参数调整注意力权重
        attn = attn * self.rescale
        # 对注意力权重进行归一化（softmax操作），使得每行权重之和为1，表示不同位置的注意力分布
        attn = attn.softmax(dim=-1)

        # 根据注意力权重对值进行加权求和，得到多头自注意力的输出，形状为[b, heads, d, hw]
        x = attn @ v
        # 对输出进行维度转置，变为[b, hw, heads, d]
        x = x.permute(0, 3, 1, 2)
        # 将多头自注意力的输出维度进行调整，合并多头维度，变为[b, h*w, num_heads * dim_head]
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)

        # 通过线性投影层将多头自注意力的输出映射回原始特征维度，再调整形状为[b, h, w, c]
        out_c = self.proj(x).view(b, h, w, c)

        # 对输入特征的原始值表示（v_inp）进行维度调整，以便通过位置嵌入模块添加位置信息，然后再调整回原始形状
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 将经过多头自注意力计算的输出与位置嵌入的输出进行相加，得到最终的输出特征
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        """
        初始化FeedForward类的实例。
        参数：
        dim (int): 同类定义中的dim参数，传入输入特征通道维度大小。
        mult (int): 同类定义中的mult参数，传入通道扩展倍数。
        """
        super().__init__()
        # 提取中间层扩展后的通道数，方便后续多处使用，增强代码可读性
        expanded_dim = dim * mult
        self.conv1 = nn.Conv2d(dim, expanded_dim, 1, 1, bias=False)
        self.activation1 = GELU()
        self.conv2 = nn.Conv2d(expanded_dim, expanded_dim, 3, 1, 1, bias=False, groups=expanded_dim)
        self.activation2 = GELU()
        self.conv3 = nn.Conv2d(expanded_dim, dim, 1, 1, bias=False)
        self.net = nn.Sequential(
            self.conv1,
            self.activation1,
            self.conv2,
            self.activation2,
            self.conv3
        )
    def forward(self, x):
        """
        前向传播函数，执行输入特征通过各层网络的处理流程并返回处理后的结果。
        参数：
        x (torch.Tensor): 输入的特征张量，形状为[b, h, w, c]，其中b为批量大小，h、w为特征图高度和宽度，c为通道维度。
        返回：
        torch.Tensor: 经过前馈神经网络处理后的输出特征张量，形状为[b, h, w, c]，与输入维度一致。
        """
        # 对输入特征张量进行维度重排，将通道维度前置，方便卷积操作
        x = x.permute(0, 3, 1, 2)
        # 通过定义好的网络模块依次处理特征张量
        out = self.net(x)
        # 再将处理后的特征张量维度重排回原始顺序，通道维度后置
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        """
        初始化MSAB类的实例。
        参数：
        dim (int): 同类定义中的dim参数，传入输入特征维度大小。
        dim_head (int): 同类定义中的dim_head参数，传入每个头的特征维度大小。
        heads (int): 同类定义中的heads参数，传入头的数量。
        num_blocks (int): 同类定义中的num_blocks参数，传入子模块组的数量。
        """
        super().__init__()
        self.blocks = nn.ModuleList([])
        # 循环构建指定数量的子模块组
        for _ in range(num_blocks):
            # 对于每个子模块组，创建包含多头自注意力模块（MS_MSA）和前馈神经网络模块（FeedForward）的列表，
            # 并且都使用PreNorm进行归一化处理，然后将这个列表添加到self.blocks中，形成ModuleList结构
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, MS_MSA(dim=dim, dim_head=dim_head, heads=heads)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, mask):
        """
        前向传播函数，按照顺序依次通过多个子模块组对输入特征进行处理，
        每个子模块组内先进行多头自注意力计算然后进行前馈神经网络处理，
        并且在每个子模块的输出上都采用残差连接的方式与输入相加，最终输出处理后的特征张量。
        参数：
        x (torch.Tensor): 输入的特征张量，形状为[b, c, h, w]，其中b表示批量大小，c表示特征维度，h、w分别表示特征图的高度和宽度。
        mask (torch.Tensor): 掩码张量，形状为[b, c, h, w]，用于在多头自注意力模块中引导特征处理过程，提供额外的约束信息。
        返回：
        torch.Tensor: 经过多个子模块组处理后的输出特征张量，形状同样为[b, c, h, w]，与输入特征张量维度保持一致。
        """
        # 首先对输入特征张量的维度进行重排，将通道维度放到最后，变为[b, h, w, c]，方便后续模块处理
        x = x.permute(0, 2, 3, 1)
        # 遍历每个子模块组
        for (attn, ff) in self.blocks:
            # 通过多头自注意力模块（attn）进行处理，传入重排后的特征张量x和对应的掩码张量mask，
            # 并且采用残差连接的方式将输出与输入x相加，更新x的值
            x = attn(x, mask=mask.permute(0, 2, 3, 1)) + x
            # 通过前馈神经网络模块（ff）进行处理，同样采用残差连接的方式将输出与输入x相加，进一步更新x的值
            x = ff(x) + x
        # 最后对处理后的特征张量x的维度进行重排，将通道维度还原到原来的位置，变为[b, c, h, w]，作为最终输出
        out = x.permute(0, 3, 1, 2)
        return out


class MST(nn.Module):
    def __init__(self, dim=28, stage=3, num_blocks=[2,2,2]):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(28, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, mask=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if mask == None:
            mask = torch.zeros((1,28,256,310)).cuda()

        # Embedding
        fea = self.lrelu(self.embedding(x))

        # Encoder
        fea_encoder = []
        masks = []
        for (MSAB, FeaDownSample, MaskDownSample) in self.encoder_layers:
            fea = MSAB(fea, mask)
            masks.append(mask)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            mask = MaskDownSample(mask)

        # Bottleneck
        fea = self.bottleneck(fea, mask)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            mask = masks[self.stage - 1 - i]
            fea = LeWinBlcok(fea, mask)

        # Mapping
        out = self.mapping(fea) + x

        return out






















