import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmodule import Conv2d_BN_AC, PRNResBlock, ConvTranspose2d_BN_AC


class InitPRN(nn.Module):
    def __init__(self):
        super(InitPRN, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size
        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=3, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=3, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=3, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=3, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=3, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=3, stride=1, with_conv_shortcut=False),
        )
        self.decoder = nn.Sequential(
            # output_padding = stride-1
            # padding=(kernelsize-1)//2
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=3, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=3, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=3, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=3, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=3, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=3, stride=1, activation=nn.Sigmoid())
        )

    def forward(self, inpt):
        x = self.layer0(inpt)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TorchNet:

    def __init__(self,
                 gpu_num=1,
                 visible_gpus='0',
                 loss_function='frse',
                 learning_rate=1e-3
                 ):
        self.gpu_num = gpu_num
        gpus = visible_gpus.split(',')
        self.visible_devices = [int(i) for i in gpus]

        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def buildInitPRN(self):
        self.model = InitPRN()

        if self.gpu_num > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.visible_devices)
        self.model.to(self.device)
        # model.cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
