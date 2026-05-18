# PAD Bancaire — Détection d’attaques de présentation faciale

Ce projet implémente un prototype de détection d’attaques de présentation faciale
dans un contexte bancaire. Le système analyse une vidéo courte d’un utilisateur et
retourne une décision métier : ACCEPT, RETRY ou REJECT.

## Solution finale

La solution finale repose sur une architecture vidéo multimodale V6 :

- validation de l’entrée vidéo ;
- challenge actif anti-replay ;
- liveness actif basé sur MediaPipe ;
- modèle vidéo CNN-LSTM avec features comportementales et rPPG ;
- classifieur behavior-pose complémentaire ;
- fusion score-level ;
- politique bancaire quality-aware.

## Pipeline runtime

Vidéo utilisateur
→ validation entrée
→ challenge actif
→ liveness actif
→ modèle PAD V6
→ score qualité vidéo
→ décision bancaire
→ ACCEPT / RETRY / REJECT

## Structure du projet

- `src/pad_system/` : système final utilisé par l’API et la démo.
- `src/behavior/` : extraction des features comportementales.
- `src/deep_learning/` : modèles, datasets et métriques.
- `experiments/03_final_models/` : modèles finaux retenus.
- `experiments/04_perspectives/` : expérimentations avancées non retenues pour le runtime.
- `reports/` : résultats, graphiques et analyses.
- `tests/` : tests de validation et démo jury.

## Modèles finaux utilisés

- Modèle vidéo principal V6 :
  `experiments/03_final_models/video_v6_behavior_pose/mixed_casia_axon_local_msu_gated_hard_balanced_rppg_v3/seed42/best_model.pth`

- Modèle behavior-pose V6 :
  `experiments/03_final_models/banking_demo_models/final_models/behavior_pose_v6/behavior_pose_clf.pkl`

- Modèle MediaPipe :
  `models/face_landmarker.task`

## Lancer les vérifications

```bash
python -m py_compile src/pad_system/video_model.py
python -m py_compile src/pad_system/router.py
python -m py_compile src/pad_system/banking_adapter.py
python -m py_compile src/pad_system/challenge_manager.py
python -m py_compile src/pad_system/input_validator.py
python -m py_compile src/pad_system/active_liveness_system.py


```
## Lancer la démo
```bash
python tests/demo_jury_video_challenge_flow.py


```


