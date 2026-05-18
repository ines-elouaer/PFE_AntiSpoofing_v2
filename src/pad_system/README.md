
#### 2. `src/pad_system/README.md`

Objectif : expliquer la partie finale utilisée en production/démo.


## Rôle du dossier

`pad_system` représente la couche d’orchestration runtime. Elle reçoit une vidéo,
valide l’entrée, vérifie le challenge actif, exécute le liveness, appelle le modèle PAD
et applique la politique bancaire.

## Fichiers principaux

- `router.py` : routeur principal du système.
- `video_model.py` : chargement du modèle vidéo V6 et calcul du score final.
- `banking_adapter.py` : transformation du score PAD en décision métier.
- `challenge_manager.py` : création, validation et expiration des challenges.
- `input_validator.py` : validation des fichiers vidéo.
- `active_liveness_system.py` : vérification des actions demandées à l’utilisateur.

## Décisions possibles

- `ACCEPT` : vidéo acceptée comme utilisateur réel.
- `RETRY` : vidéo ambiguë ou qualité insuffisante.
- `REJECT` : attaque probable.

## Remarque

Ce dossier est la partie la plus importante pour le déploiement API.