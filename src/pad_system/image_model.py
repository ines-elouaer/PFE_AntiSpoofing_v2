import os
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD


class ImagePADModel:
    """
    Branche image du système PAD bancaire.

    Principe :
    - entrée : image JPG/PNG
    - preprocessing : Resize + Normalize
    - transformation : image statique répétée T fois
    - modèle : CNN+LSTM fine-tuné sur CelebA-Spoof
    - sortie : score spoof entre 0 et 1
    """

    def __init__(
        self,
        checkpoint_path: str = r"E:\PFE_AntiSpoofing_v2\experiments\celeba_finetune_v3_8000_unfreeze\seed42\best_model_celeba_ft.pth",
        img_size: int = 224,
        seq_len: int = 16,
    ):
        self.checkpoint_path = checkpoint_path
        self.img_size = img_size
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[IMAGE] Device: {self.device}")
        print(f"[IMAGE] Loading checkpoint: {self.checkpoint_path}")

        self.model = self._load_model()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model(self):
        ckpt_path = Path(self.checkpoint_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint image introuvable: {ckpt_path}")

        # Dans ton fine-tuning CelebA, le fichier best_model_celeba_ft.pth
        # est souvent sauvegardé comme state_dict direct.
        state = torch.load(str(ckpt_path), map_location=self.device)

        # Si le checkpoint est un dict complet, on récupère le state_dict.
        if isinstance(state, dict) and "model_state" in state:
            state_dict = state["model_state"]
            cfg = state.get("config", {})
        elif isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
            cfg = state.get("config", {})
        else:
            state_dict = state
            cfg = {}

        # Config par défaut compatible avec ton pipeline CelebA fine-tuning.
        model = CNN_LSTM_PAD(
            hidden=cfg.get("hidden", 256),
            num_layers=cfg.get("num_layers", 1),
            bidir=cfg.get("bidir", False),
            lstm_dropout=cfg.get("lstm_dropout", 0.0),
            head_dropout=0.0,
            pretrained_backbone=False,
            temporal_pool=cfg.get("temporal_pool", "median"),
            use_behav=cfg.get("use_behav", True),
            behav_dim=cfg.get("behav_dim", 9),
            behav_hidden=cfg.get("behav_hidden", 16),
        )

        # Corriger DataParallel éventuel : "module.xxx" → "xxx"
        if isinstance(state_dict, dict) and all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)

        print("[IMAGE] Checkpoint chargé avec succès.")
        return model

    @torch.no_grad()
    def predict(self, image_path: str) -> float:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image introuvable: {image_path}")

        print(f"[INFO] Analyse image: {image_path}")

        img = Image.open(image_path).convert("RGB")
        frame = self.transform(img)  # [3, 224, 224]

        # Ton modèle CelebA utilise CNN+LSTM :
        # on répète donc l'image T fois pour créer une séquence statique.
        seq = frame.unsqueeze(0).repeat(self.seq_len, 1, 1, 1)  # [T, 3, 224, 224]
        seq = seq.unsqueeze(0).to(self.device)                  # [1, T, 3, 224, 224]

        # Pas de comportement réel pour image statique.
        behav = torch.zeros((1, 9), dtype=torch.float32).to(self.device)

        logits = self.model(seq, behav)
        proba = torch.softmax(logits, dim=1)

        # Classe 1 = spoof / attaque
        spoof_score = proba[0, 1].item()

        return float(spoof_score)