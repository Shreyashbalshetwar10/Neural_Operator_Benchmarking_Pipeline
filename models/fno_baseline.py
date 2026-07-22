import torch
import torch.nn as nn
import torch.nn.functional as F

r"The input channels are : mask, re_field, sin(aoa_field), cos(aoa_field) and the output channels are : u,v,p"

class AttentionSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, max_modes):
        super(AttentionSpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes = max_modes

        # Standard FNO complex weights
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, max_modes, max_modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, max_modes, max_modes, dtype=torch.cfloat))

        hidden_dim = in_channels * 2
        
        self.attention_cnn = nn.Sequential(
            # Input shape: (Batch, in_channels, max_modes, max_modes)
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            # Output shape: (Batch, in_channels, max_modes, max_modes)
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):

        batchsize = x.shape[0]

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        x_ft_c1 = x_ft[:, :, :self.max_modes, :self.max_modes]

        amp_c1 = torch.abs(x_ft_c1) 

        attention_scores_c1 = self.attention_cnn(amp_c1) 

        x_ft_c1_attended = x_ft_c1 * (1 + attention_scores_c1)

        out_ft[:, :, :self.max_modes, :self.max_modes] = \
            torch.einsum("bixy,ioxy->boxy", x_ft_c1_attended, self.weights1)  

        x_ft_c2 = x_ft[:, :, -self.max_modes:, :self.max_modes]
        amp_c2 = torch.abs(x_ft_c2)
        
        attention_scores_c2 = self.attention_cnn(amp_c2)

        x_ft_c2_attended = x_ft_c2 * (1 + attention_scores_c2)

        out_ft[:, :, -self.max_modes:, :self.max_modes] = \
            torch.einsum("bixy,ioxy->boxy", x_ft_c2_attended, self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class FNO2d(nn.Module): 
    def __init__(self, modes, width, num_in_channels=3, num_out_channels=3):
        super(FNO2d, self).__init__()

        self.modes = modes
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(num_in_channels, self.width) # input channel is num_in_channels: (u, v, p)

        self.conv0 = AttentionSpectralConv2d(self.width, self.width, self.modes)
        self.conv1 = AttentionSpectralConv2d(self.width, self.width, self.modes)
        self.conv2 = AttentionSpectralConv2d(self.width, self.width, self.modes)
        self.conv3 = AttentionSpectralConv2d(self.width, self.width, self.modes)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_out_channels)

    def forward(self, x):
        batchsize = x.shape[0]
        channels = x.shape[1]
        size_x = x.shape[2]
        size_y = x.shape[3]

        x = F.pad(x, [0,self.padding, 0,self.padding])

        x = x.permute(0,2,3,1)
        x = self.fc0(x)

        x1 = self.conv0(x.permute(0,3,1,2)) 
        x2 = self.w0(x.permute(0,3,1,2))
        x = F.gelu(x1 + x2).permute(0,2,3,1)

        x1 = self.conv1(x.permute(0,3,1,2)) 
        x2 = self.w1(x.permute(0,3,1,2))
        x = F.gelu(x1 + x2).permute(0,2,3,1)

        x1 = self.conv2(x.permute(0,3,1,2)) 
        x2 = self.w2(x.permute(0,3,1,2))
        x = F.gelu(x1 + x2).permute(0,2,3,1)

        x1 = self.conv3(x.permute(0,3,1,2)) 
        x2 = self.w3(x.permute(0,3,1,2))
        x = F.gelu(x1 + x2).permute(0,2,3,1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)

        return x[:, :, :size_x, :size_y] 
    
class UnitGaussianNormalizer(object):
    def __init__(self, mask, y, eps=1e-5):
        super().__init__()

        self.eps = eps

        if mask.ndim == 4 and mask.shape[1] > 1:
            mask = mask[:, 0:1, :, :]

        # channel 0:
        # 1 = solid
        # 0 = fluid

        fluid_mask = (mask < 0.5).bool()

        means = []
        stds = []

        # ONLY normalize physical channels
        # channels: Ux, Uy, p
        for c in range(y.shape[1]):

            channel_data = y[:, c:c+1, :, :]

            valid_pixels = torch.masked_select(channel_data, fluid_mask)
            if valid_pixels.numel() == 0:
                raise ValueError(f"[FATAL] Channel {c}: valid_pixels is empty! The mask is blocking the entire dataset.")
            
            if torch.isnan(valid_pixels).any() or torch.isinf(valid_pixels).any():
                print(f"[WARNING] NaN or Inf detected in raw dataset for Channel {c}! Filtering them out to save normalizer...")
                # Filter out the poison so the mean doesn't become NaN
                valid_pixels = valid_pixels[~torch.isnan(valid_pixels)]
                valid_pixels = valid_pixels[~torch.isinf(valid_pixels)]

            means.append(valid_pixels.mean())
            stds.append(valid_pixels.std())

        self.mean = torch.stack(means).view(1, -1, 1, 1)
        self.std  = torch.stack(stds).view(1, -1, 1, 1)

    def encode(self, mask, to_normalise):

        fluid_mask = (mask < 0.5).float()

        x_norm = to_normalise.clone()

        x_norm = (
            to_normalise - self.mean
        ) / (self.std + self.eps)

        # zero-out solid regions
        x_norm *= fluid_mask.float()

        return x_norm

    def decode(self, to_denormalise):

        x_dec = to_denormalise.clone()

        x_dec = (
            to_denormalise * (self.std + self.eps)
        ) + self.mean

        return x_dec

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
