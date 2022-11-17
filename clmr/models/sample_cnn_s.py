import torch
import torch.nn as nn
from .model import Model


class SampleCNNS(Model):
    def __init__(self, strides, supervised, out_dim):
        super(SampleCNNS, self).__init__()

        self.strides = strides
        self.supervised = supervised
        sequential_ = [
            nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        ]

        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
        ]

        assert len(self.hidden) == len(
            self.strides
        ), "Number of hidden layers and strides are not equal"
        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            sequential_.append(
                nn.Sequential(
                    nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride),
                )
            )

        # 1 x 512
        sequential_.append(
            nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            )
        )

        self.sequential = nn.Sequential(*sequential_)

        if self.supervised:
            self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        out = self.sequential(x)
        if self.supervised:
            out = self.dropout(out)

        out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)
        return logit
