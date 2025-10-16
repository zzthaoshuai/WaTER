import torch
import torch.nn as nn
import numpy as np
import pywt
import math
from torch.autograd import Function

from loss import TSandDSLoss


class DWT(nn.Module):
    """
    input: the 3D data to be decomposed -- (N, C, D, H, W)
    output: lfc -- (N, C, D/2, H/2, W/2)
            hfc_llh -- (N, C, D/2, H/2, W/2)
            hfc_lhl -- (N, C, D/2, H/2, W/2)
            hfc_lhh -- (N, C, D/2, H/2, W/2)
            hfc_hll -- (N, C, D/2, H/2, W/2)
            hfc_hlh -- (N, C, D/2, H/2, W/2)
            hfc_hhl -- (N, C, D/2, H/2, W/2)
            hfc_hhh -- (N, C, D/2, H/2, W/2)
    """

    def __init__(self, wavename):
        """
        3D discrete wavelet transform (DWT) for 3D data decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):

        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(
            self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(
            self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1):end]

        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, input):
        """
        :param input: the 3D data to be decomposed
        :return: the eight components of the input data, one low-frequency and seven high-frequency components
        """
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                    self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)

class DWTFunction(Function):
    @staticmethod
    def forward(ctx, input,
                matrix_Low_0, matrix_Low_1, matrix_Low_2,
                matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)

        LL = torch.matmul(L, matrix_Low_1).transpose(dim0=2, dim1=3)
        LH = torch.matmul(L, matrix_High_1).transpose(dim0=2, dim1=3)
        HL = torch.matmul(H, matrix_Low_1).transpose(dim0=2, dim1=3)
        HH = torch.matmul(H, matrix_High_1).transpose(dim0=2, dim1=3)

        LLL = torch.matmul(matrix_Low_2, LL).transpose(dim0=2, dim1=3)
        LLH = torch.matmul(matrix_Low_2, LH).transpose(dim0=2, dim1=3)
        LHL = torch.matmul(matrix_Low_2, HL).transpose(dim0=2, dim1=3)
        LHH = torch.matmul(matrix_Low_2, HH).transpose(dim0=2, dim1=3)
        HLL = torch.matmul(matrix_High_2, LL).transpose(dim0=2, dim1=3)
        HLH = torch.matmul(matrix_High_2, LH).transpose(dim0=2, dim1=3)
        HHL = torch.matmul(matrix_High_2, HL).transpose(dim0=2, dim1=3)
        HHH = torch.matmul(matrix_High_2, HH).transpose(dim0=2, dim1=3)

        LF = LLL
        HF = torch.cat((LLH, LHL, LHH, HLL, HLH, HHL), dim=1)

        return LF, HF

    @staticmethod
    def backward(ctx, grad_LLL, grad_LLH, grad_LHL, grad_LHH,
                 grad_HLL, grad_HLH, grad_HHL, grad_HHH):
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_LL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLL.transpose(dim0=2, dim1=3)), torch.matmul(
            matrix_High_2.t(), grad_HLL.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_LH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLH.transpose(dim0=2, dim1=3)), torch.matmul(
            matrix_High_2.t(), grad_HLH.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_HL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHL.transpose(dim0=2, dim1=3)), torch.matmul(
            matrix_High_2.t(), grad_HHL.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_HH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHH.transpose(dim0=2, dim1=3)), torch.matmul(
            matrix_High_2.t(), grad_HHH.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()),
                           torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()),
                           torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(
            matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None, None, None, None, None

class PatchEmbedding(nn.Module):

    def __init__(self, patch_size, img_size, in_channels, out_channels):
        super().__init__()
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.patch_embeddings = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, out_channels))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, out_channels, D', H', W')
        x = x.flatten(2)  # (B, out_channels, n_patches)
        x = x.transpose(-1, -2)  # (B, n_patches, out_channels)
        embeddings = x + self.position_embeddings  # (B, n_patches, out_channels)
        embeddings = self.dropout(embeddings)

        return embeddings

class PoolEmbedding(nn.Module):

    def __init__(self, patch_size, img_size, in_channels):
        super().__init__()
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.pool_embedding = nn.AvgPool3d(kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool_embedding(x)  # (B, in_channels, D', H', W')
        x = x.flatten(2)  # (B, in_channels, n_patches)
        x = x.transpose(-1, -2)  # (B, n_patches, in_channels)
        embeddings = x + self.position_embeddings  # (B, n_patches, in_channels)
        embeddings = self.dropout(embeddings)

        return embeddings

class WCSA(nn.Module):
    """
    Efficient Paired Attention Block, modified for Wavelet Cross-Attention.
    """

    def __init__(self,  in_channels, hidden_size, num_heads=4, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.1)
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.1)

        self.q_c = nn.Linear(in_channels, in_channels, bias=False)
        self.q_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_s = nn.Linear(in_channels // 4, in_channels // 4, bias=False)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

    def forward(self, s, h, sh):
        """
        Args:
            h: Query (B, N, C)
            sh: Concatenated Key-Value (B, N, C)
        """
        B, N, C = sh.shape
        C_c = s.shape[2]
        C_s = h.shape[2]

        # Project Q and KV
        q_c = self.q_c(s).reshape(B, self.num_heads, N, C_c // self.num_heads).transpose(-1, -2)
        q_s = self.q_s(sh).reshape(B, self.num_heads, N, C // self.num_heads)
        k_c = self.k_c(sh).reshape(B, self.num_heads, N, C // self.num_heads)
        v_c = self.v_c(sh).reshape(B, self.num_heads, N, C // self.num_heads)
        k_s = self.k_s(sh).reshape(B, self.num_heads, N, C // self.num_heads)
        v_s = self.v_s(h).reshape(B, self.num_heads, N, C_s // self.num_heads)


        # Channel Attention
        attn_ca = (q_c @ k_c) / math.sqrt(k_c.shape[-1]) * self.temperature
        attn_ca = attn_ca.softmax(dim=-1)
        attn_ca = self.attn_drop(attn_ca)
        x_ca = (attn_ca @ v_c.transpose(-2, -1)).reshape(B, N, C_c)

        # Spatial Attention
        attn_sa = (q_s @ k_s.transpose(-2, -1)) / math.sqrt(k_s.shape[-1]) * self.temperature2
        attn_sa = attn_sa.softmax(dim=-1)
        attn_sa = self.attn_drop_2(attn_sa)
        x_sa = (attn_sa @ v_s).reshape(B, N, C_s)

        x = torch.cat([x_ca, x_sa], dim=2)

        return x

class TransformerBlock(nn.Module):
    """
    A transformer block, modified for Wavelet Cross-Attention.
    """

    def __init__(
            self,
             in_channels: int,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.1,
            patch_size = (4, 4, 4),
            img_size = (64, 64, 64),
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.patch_embedding = PatchEmbedding(patch_size=patch_size, img_size=img_size, in_channels=6, out_channels=in_channels//4)
        self.pool_embedding = PoolEmbedding(patch_size=patch_size, img_size=img_size, in_channels=in_channels)
        self.norm_s = nn.LayerNorm(in_channels)
        self.norm_h = nn.LayerNorm(in_channels // 4)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.wcsa_block = WCSA(
            in_channels= in_channels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            channel_attn_drop=dropout_rate,
            spatial_attn_drop=dropout_rate,
        )
        self.conv = nn.Sequential(nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1), nn.BatchNorm3d(hidden_size), nn.LeakyReLU(inplace=True))
        self.recon = nn.ConvTranspose3d(hidden_size, hidden_size, kernel_size=patch_size[0], stride=patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size


    def forward(self, s, h):
        """
        Args:
            s: Skip connection from encoder. (B, C_s, H, W, D)
            h: High-frequency wavelet component. (B, C_h, H, W, D)
        """

        s = self.pool_embedding(s)
        h = self.patch_embedding(h)

        s = self.norm_s(s)
        h = self.norm_h(h)

        # Concatenate s and h for KV
        sh = torch.cat([s, h], dim=2)  # (B, N, C_h + C_s)

        B, N, C = sh.shape
        H, W, D = (self.img_size[0] // self.patch_size[0],
                   self.img_size[1] // self.patch_size[1],
                   self.img_size[2] // self.patch_size[2])

        # Compute attention using h as Q and kv as KV
        attn = sh + self.wcsa_block(s, h, sh) * self.gamma

        # Reshape back to (B, C, H, W, D)
        attn = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        x = self.conv(attn)

        x = self.recon(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x (torch.Tensor): (batch_size, in_channels, D, H, W)

        Returns:
            tuple:
                - s (torch.Tensor): (batch_size, out_channels, D, H, W)
                - p (torch.Tensor): (batch_size, out_channels, D//2, H//2, W//2)
        """
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: list, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels[0], in_channels[0], kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels[0] + in_channels[1] + in_channels[2], out_channels)

    def forward(self, x: torch.Tensor, a:torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch_size, in_channels, D, H, W)
            s (torch.Tensor): (batch_size, skip_channels, D, H, W)

        Returns:
            torch.Tensor: (batch_size, out_channels, D, H, W)
        """
        x = self.up(x)
        x = torch.cat([x, a, s], dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        # Wavelet transform
        self.dwt = DWT("haar")

        # Encoder
        self.e1 = EncoderBlock(1, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)

        # Transformer attention modules
        self.attention1 = TransformerBlock(64, hidden_size=80, num_heads=4,
                                           patch_size=(8, 8, 8), img_size=(32, 32, 32))
        self.attention2 = TransformerBlock(128, hidden_size=160, num_heads=4,
                                           patch_size=(4, 4, 4), img_size=(16, 16, 16))
        self.attention3 = TransformerBlock(256, hidden_size=320, num_heads=4,
                                           patch_size=(2, 2, 2), img_size=(8, 8, 8))

        # Bottleneck
        self.b1 = ConvBlock(256, 512)

        # Decoder
        self.d1 = DecoderBlock([512, 320, 256], 256)
        self.d2 = DecoderBlock([256, 160, 128], 128)
        self.d3 = DecoderBlock([128, 80, 64], 64)
        self.d4 = DecoderBlock([64, 32, 32], 32)

        # Output layer
        self.output = nn.Conv3d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, 1, D, H, W)

        Returns:
            torch.Tensor: Segmentation mask (B, 1, D, H, W)
        """
        # Wavelet decomposition
        l1, h1 = self.dwt(x)
        l2, h2 = self.dwt(l1)
        l3, h3 = self.dwt(l2)

        # Encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Wavelet cross-scale attention
        a1 = self.attention1(s2, h1)
        a2 = self.attention2(s3, h2)
        a3 = self.attention3(s4, h3)

        # Bottleneck
        b1 = self.b1(p4)

        # Decoder
        d1 = self.d1(b1, a3, s4)
        d2 = self.d2(d1, a2, s3)
        d3 = self.d3(d2, a1, s2)
        d4 = self.d4(d3, s1, s1)

        # Output
        out = self.output(d4)
        # out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    x = torch.randn((1, 1, 64, 64, 64)).cuda()
    model = UNet3D().cuda()
    output = model(x)
    print(output.shape)
