# src/deep_learning/models_cnn_lstm.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class MobileNetV3Feature(nn.Module):

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = mobilenet_v3_large(weights=weights)

        self.features = m.features
        self.avgpool = m.avgpool
        self.out_dim = m.classifier[0].in_features  # typically 960

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] -> [B,F]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class CNN_LSTM_PAD(nn.Module):
    
    def __init__(
        self,
        hidden: int = 256,
        num_layers: int = 1,
        bidir: bool = False,
        lstm_dropout: float = 0.0,
        head_dropout: float = 0.2,
        pretrained_backbone: bool = True,
        temporal_pool: str = "mean",  
    ):
        super().__init__()
        self.backbone = MobileNetV3Feature(pretrained=pretrained_backbone)
        feat_dim = self.backbone.out_dim

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        self.temporal_pool = temporal_pool
        out_dim = hidden * (2 if bidir else 1)

        self.head = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, 2),
        )

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,C,H,W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.backbone(x)          # [B*T, F]
        feats = feats.view(B, T, -1)      # [B, T, F]

        out, _ = self.lstm(feats)         

        if self.temporal_pool == "mean":
            pooled = out.mean(dim=1)      
        elif self.temporal_pool == "median":
            pooled = out.median(dim=1).values  
        elif self.temporal_pool == "last":
            pooled = out[:, -1, :]
        else:
            raise ValueError(f"Invalid temporal_pool: {self.temporal_pool}")

        return self.head(pooled)
