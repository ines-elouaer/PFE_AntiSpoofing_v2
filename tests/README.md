# Tests et démonstrations

Ce dossier contient les scripts utilisés pour valider le système.

## Scripts principaux

- `test_challenge_manager.py` : teste la création, validation et réutilisation des challenges.
- `test_input_validation.py` : teste la validation des vidéos, images et fichiers invalides.
- `test_model_direct_video.py` : teste directement le modèle vidéo.
- `demo_jury_video_challenge_flow.py` : démo complète avec webcam.

## Démo principale

```bash
python tests/demo_jury_video_challenge_flow.py
```
La démo suit le flux :

challenge/start
→ capture webcam
→ validation vidéo
→ liveness actif
→ modèle PAD V6
→ décision finale