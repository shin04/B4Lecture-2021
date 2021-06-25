import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from conformer import ConformerBlock
# from torchinfo import summary


class SpecAugBlock(nn.Module):
    def __init__(self, sr: int, win_size: int, overlap: int, n_mels: int, training: bool = True):
        super(SpecAugBlock, self).__init__()

        self.training = training

        # # spectrogram
        # self.spectrogram_extractor = Spectrogram(
        #     n_fft=win_size, hop_length=int(win_size*overlap), win_length=win_size)

        # # Logmel feature extractor
        # self.logmel_extractor = LogmelFilterBank(
        #     sr=sr, n_fft=win_size, n_mels=n_mels)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=2, time_stripes_num=2, freq_drop_width=2, freq_stripes_num=2)

        self.batch_norm = nn.BatchNorm2d(81)

    def forward(self, input):
        """
        input (batch_size, data_length)
        """

        # x = self.spectrogram_extractor(input)
        # x = self.logmel_extractor(x)
        x = input

        x = x.transpose(1, 3)
        x = self.batch_norm(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2)):

        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu_(x)
        x = F.avg_pool2d(x, kernel_size=pool_size)

        return x


class ConformerModel(nn.Module):
    def __init__(self, training=True):
        super(ConformerModel, self).__init__()

        self.training = training

        self.spec_aug_block = SpecAugBlock(22050, int(22050*0.025), 0.4, 64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=128)
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(128*16*40, 128, bias=True)

        self.conf_block1 = ConformerBlock(
            dim=128,
            dim_head=32,
            heads=4,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
        )

        self.conf_block2 = ConformerBlock(
            dim=128,
            dim_head=32,
            heads=4,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
        )
        self.conf_block3 = ConformerBlock(
            dim=128,
            dim_head=32,
            heads=4,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
        )

        self.flatten2 = nn.Flatten()
        self.fc2 = nn.Linear(128, 256, bias=True)
        self.output_layer = nn.Linear(256, 10, bias=True)

    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """

        x = self.spec_aug_block(input)
        # x = input

        x = self.conv_block1(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.flatten1(x)
        x = self.fc1(x)

        x = x.unsqueeze(1)
        x = self.conf_block1(x)
        x = self.conf_block2(x)
        x = self.conf_block3(x)

        x = self.flatten2(x)
        x = self.fc2(x)
        output = self.output_layer(x)

        return output


class GRUModel(nn.Module):
    def __init__(self, training=True):
        super(GRUModel, self).__init__()

        self.training = training

        self.spec_aug_block = SpecAugBlock(22050, int(22050*0.025), 0.4, 64)

        self.rnn_layer1 = nn.GRU(81, 256, 2, dropout=0.2)
        self.flatten1 = nn.Flatten()
        self.output_layer = nn.Linear(32*256, 10, bias=True)

    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """

        x = self.spec_aug_block(input)
        x = x.squeeze(1)

        x = self.rnn_layer1(x)[0]
        x = self.flatten1(x)
        output = self.output_layer(x)

        return output


if __name__ == '__main__':
    batch_audio = torch.empty(32, 1, 100, 16).uniform_(-1, 1).cuda()

    model = ConformerModel().cuda()
    # summary(model, input=(1, 1, 22050*1))
    print(model(batch_audio))
