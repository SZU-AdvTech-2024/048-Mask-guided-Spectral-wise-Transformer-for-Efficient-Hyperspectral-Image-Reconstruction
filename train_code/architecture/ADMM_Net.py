import torch
import torch.nn as nn
import torch.nn.functional as F


def A(x, Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp*Phi
    return x

class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)

        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        """
        Input: torch.Size([1, 26, 250, 250])
        dconv1: torch.Size([1, 32, 256, 256])
        maxpool1: torch.Size([1, 32, 128, 128])
        dconv2: torch.Size([1, 64, 128, 128])
        maxpool2: torch.Size([1, 64, 64, 64])
        dconv3: torch.Size([1, 128, 64, 64])
        Upsample2: torch.Size([1, 64, 128, 128])
        cat1: torch.Size([1, 128, 128, 128])
        dconv_up2: torch.Size([1, 64, 128, 128])
        Upsample1: torch.Size([1, 32, 256, 256])
        cat2: torch.Size([1, 64, 256, 256])
        dconv_up1: torch.Size([1, 32, 256, 256])
        conv_last: torch.Size([1, 26, 256, 256])
        afn_last: torch.Size([1, 26, 256, 256])
        output: torch.Size([1, 26, 256, 256])
        """
        b, c, h_inp, w_inp = x.shape  # b for batch_size, c for channal_num, h_inp&w_inp for height&width
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')  # 将原图像填充成可被(hb, wb)整除的大小
        inputs = x
        conv1 = self.dconv_down1(x)  # 两次卷积，图像宽高不变, c:32
        x = self.maxpool(conv1)  # 下采样，图像宽高减半

        conv2 = self.dconv_down2(x)  # c:64
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)  # c:128

        x = self.upsample2(conv3)  # 一次反卷积，feature_map宽高翻倍
        x = torch.cat([x, conv2], dim=1)  # c:128

        x = self.dconv_up2(x)  # c:64
        x = self.upsample1(x)  # 一次反卷积
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs  # 残差连接

        return out[:, :, :h_inp, :w_inp]

def shift_3d(inputs,step=2):
    """
    对每个通道的向量右移指定的步长
    """
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    """
    对每个通道的向量左移指定的步长
    """
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

class ADMM_net(nn.Module):

    def __init__(self):
        super(ADMM_net, self).__init__()
        # 9阶段的展开ADMM
        self.unet1 = Unet(28, 28)
        self.unet2 = Unet(28, 28)
        self.unet3 = Unet(28, 28)
        self.unet4 = Unet(28, 28)
        self.unet5 = Unet(28, 28)
        self.unet6 = Unet(28, 28)
        self.unet7 = Unet(28, 28)
        self.unet8 = Unet(28, 28)
        self.unet9 = Unet(28, 28)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma3 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma4 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma5 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma6 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma7 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma8 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma9 = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, y, input_mask=None):
        if input_mask == None:
            Phi = torch.rand((1, 28, 256, 310)).cuda()
            Phi_s = torch.rand((1, 256, 310)).cuda()
        else:
            Phi, Phi_s = input_mask
        x_list = []
        theta = At(y, Phi)  # 原文中的辅助变量v， v = Phi_T * y
        b = torch.zeros_like(Phi)  # 原文中的辅助变量U
        ### 1-3
        yb = A(theta+b, Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma1),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet1(x1)
        theta = shift_3d(theta)
        b = b - (x-theta)
        x_list.append(theta)
        yb = A(theta+b, Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma2),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet2(x1)
        theta = shift_3d(theta)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma3),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet3(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        ### 4-6
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma4),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet4(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma5),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet5(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma6),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet6(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        ### 7-9
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma7),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet7(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma8),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet8(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma9),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet9(x1)
        theta = shift_3d(theta)
        return theta[:, :, :, 0:256]
