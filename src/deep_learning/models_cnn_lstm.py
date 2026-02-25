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
        self.out_dim = m.classifier[0].in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        head_dropout: float = 0.5,
        pretrained_backbone: bool = True,
        temporal_pool: str = "mean",
        # NEW:
        use_behav: bool = False,
        behav_dim: int = 9,
        behav_hidden: int = 32,
    ):
        super().__init__()
        self.use_behav = use_behav
        self.behav_dim = behav_dim

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

        
        if self.use_behav:
            self.behav_mlp = nn.Sequential(
                nn.Linear(behav_dim, behav_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
            )
            fusion_dim = out_dim + behav_hidden
        else:
            self.behav_mlp = None
            fusion_dim = out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, 2),
        )

    # PTS 
    def freeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_k_backbone_blocks(self, k: int = 4):
        self.freeze_all_backbone()
        feats = self.backbone.features
        n = len(feats)
        k = max(0, min(k, n))
        for i in range(n - k, n):
            for p in feats[i].parameters():
                p.requires_grad = True
        for p in self.backbone.avgpool.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, behav: torch.Tensor | None = None) -> torch.Tensor:
       
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.backbone(x)       # [B*T, F]
        feats = feats.view(B, T, -1)   # [B, T, F]

        out, _ = self.lstm(feats)      # [B, T, H]

        if self.temporal_pool == "mean":
            pooled = out.mean(dim=1)
        elif self.temporal_pool == "median":
            pooled = out.median(dim=1).values
        elif self.temporal_pool == "last":
            pooled = out[:, -1, :]
        else:
            raise ValueError(f"Invalid temporal_pool: {self.temporal_pool}")

        if self.use_behav:
            if behav is None:
                behav = torch.zeros(B, self.behav_dim, device=pooled.device, dtype=pooled.dtype)
            b = self.behav_mlp(behav)
            pooled = torch.cat([pooled, b], dim=1)

        return self.head(pooled)