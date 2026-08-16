import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out += residual
        return self.relu(out)

class SPARK_Netv2_Residual_one_model_plus0(nn.Module):
    def __init__(self, coils, kernelsize, acsx, acsy):
        super().__init__()
        self.acsx = acsx
        self.acsy = acsy
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(coils*2, coils*2, kernelsize, padding=1, bias=False)
        self.residual_block1 = ResidualBlock(coils*2)
        self.residual_block2 = ResidualBlock(coils*2)
        self.conv2 = nn.Conv2d(coils*2, coils, 1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(coils, coils, kernelsize, padding=1, bias=False)
        self.conv0 = nn.Conv2d(coils * 2, coils, kernelsize, padding=1, bias=False)

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.residual_block1(y)
        y = self.residual_block2(y)
        y = self.relu(self.conv2(y))
        out = self.conv3(y)+self.conv0(x)
        loss_out = out[:, :, self.acsx[0]:self.acsx[-1]+1, self.acsy[0]:self.acsy[-1]+1]
        return out, loss_out
