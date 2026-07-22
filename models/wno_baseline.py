import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# GPU-native 2D Wavelet Transform using Haar wavelets
# Haar is the simplest wavelet and is fully differentiable on GPU. It is used in the WNO paper.
# ==============================================================================

def haar_dwt2d(x):
    """
    Single-level 2D Haar DWT.
    Input:  (B, C, H, W)
    Output: (B, C*4, H//2, W//2)  — LL, LH, HL, HH concatenated along channel dim
    """
    # Row transform
    x_even = x[:, :, :, 0::2]
    x_odd  = x[:, :, :, 1::2]
    L = (x_even + x_odd) / 2.0   # low pass
    H = (x_even - x_odd) / 2.0   # high pass

    # Column transform on L
    LL = (L[:, :, 0::2, :] + L[:, :, 1::2, :]) / 2.0
    LH = (L[:, :, 0::2, :] - L[:, :, 1::2, :]) / 2.0

    # Column transform on H
    HL = (H[:, :, 0::2, :] + H[:, :, 1::2, :]) / 2.0
    HH = (H[:, :, 0::2, :] - H[:, :, 1::2, :]) / 2.0

    return torch.cat([LL, LH, HL, HH], dim=1)  # (B, C*4, H//2, W//2)


def haar_idwt2d(coeffs, C_out):
    """
    Single-level 2D Haar IDWT.
    Input:  (B, C*4, H//2, W//2)
    Output: (B, C, H, W)
    """
    B, _, Hh, Wh = coeffs.shape
    LL = coeffs[:, 0*C_out:1*C_out, :, :]
    LH = coeffs[:, 1*C_out:2*C_out, :, :]
    HL = coeffs[:, 2*C_out:3*C_out, :, :]
    HH = coeffs[:, 3*C_out:4*C_out, :, :]

    # Inverse column transform
    L = torch.zeros(B, C_out, Hh*2, Wh, device=coeffs.device)
    H = torch.zeros(B, C_out, Hh*2, Wh, device=coeffs.device)
    L[:, :, 0::2, :] = LL + LH
    L[:, :, 1::2, :] = LL - LH
    H[:, :, 0::2, :] = HL + HH
    H[:, :, 1::2, :] = HL - HH

    # Inverse row transform
    out = torch.zeros(B, C_out, Hh*2, Wh*2, device=coeffs.device)
    out[:, :, :, 0::2] = L + H
    out[:, :, :, 1::2] = L - H

    return out


# ==============================================================================
# Wavelet Convolution Layer
# ==============================================================================
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level=2):
        super(WaveConv2d, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.level        = level

        self.coeff_conv = nn.Conv2d(
            in_channels  * 4,
            out_channels * 4,
            kernel_size=1,
            bias=True
        )

        if level >= 2:
            self.coeff_conv2 = nn.Conv2d(
                in_channels  * 4,
                out_channels * 4,
                kernel_size=1,
                bias=True
            )

    def forward(self, x):
        B, C, H, W = x.shape

        # Level 1 DWT
        coeffs1 = haar_dwt2d(x)          # (B, C*4, H//2, W//2)

        if self.level >= 2:

            ll1    = coeffs1[:, :C, :, :]         # LL subband
            rest1  = coeffs1[:, C:, :, :]          # LH, HL, HH

            coeffs2 = haar_dwt2d(ll1)              # (B, C*4, H//4, W//4)
            coeffs2 = self.coeff_conv2(coeffs2)    # learned mixing

            ll1_rec = haar_idwt2d(coeffs2, self.out_channels)

            coeffs1_mod = self.coeff_conv(coeffs1)

            coeffs1_mod[:, :self.out_channels, :, :] = ll1_rec

            out = haar_idwt2d(coeffs1_mod, self.out_channels)

        else:
            coeffs1 = self.coeff_conv(coeffs1)
            out = haar_idwt2d(coeffs1, self.out_channels)

        return out


# ==============================================================================
# WNO Block
# ==============================================================================
class WNOBlock(nn.Module):
    def __init__(self, width, level=2):
        super(WNOBlock, self).__init__()
        self.wave_conv = WaveConv2d(width, width, level=level)
        self.bypass    = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.wave_conv(x) + self.bypass(x))


# ==============================================================================
# WNO2d Model
# ==============================================================================
class WNO2d(nn.Module):
    def __init__(self, width=32, level=2,
                 num_in_channels=6, num_out_channels=3):
        super(WNO2d, self).__init__()

        self.p      = nn.Linear(num_in_channels, width)
        self.block0 = WNOBlock(width, level)
        self.block1 = WNOBlock(width, level)
        self.block2 = WNOBlock(width, level)
        self.block3 = WNOBlock(width, level)
        self.q      = nn.Linear(width, num_out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        x = x.permute(0, 3, 1, 2)
        return x