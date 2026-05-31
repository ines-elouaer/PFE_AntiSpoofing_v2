# PAD Bancaire — Détection d’attaques de présentation faciale

## Présentation générale

Ce projet implémente un prototype de **détection d’attaques de présentation faciale** dans un contexte bancaire.

Le système analyse une courte vidéo capturée en temps réel et retourne une décision métier adaptée à un parcours d’authentification :

| Décision | Signification |
|---|---|
| `ACCEPT` | Identité vérifiée |
| `RETRY` | Nouvelle capture demandée |
| `REJECT` | Risque d’attaque détecté |

L’objectif principal est de renforcer l’authentification faciale contre les attaques PAD comme **photo**, **vidéo rejouée**, **support artificiel**...

---

## Solution finale

La solution finale repose sur une architecture vidéo multimodale **FOMA V6**, combinant plusieurs modules complémentaires :

- validation de l’entrée vidéo ;
- challenge actif anti-replay à usage unique ;
- détection de vivacité active avec MediaPipe ;
- modèle vidéo CNN-LSTM-behavior ;
- extraction de features comportementales et rPPG ;
- classifieur complémentaire behavior-pose V6 ;
- fusion score-level ;
- politique bancaire `ACCEPT / RETRY / REJECT` ;
- journalisation des résultats JSON.

---

## Pipeline runtime

```text
Vidéo utilisateur
  → Validation de l’entrée
  → Challenge actif
  → Liveness actif
  → Modèle PAD V6
  → Behavior-pose V6
  → Fusion des scores
  → Qualité vidéo
  → Décision bancaire
  → ACCEPT / RETRY / REJECT
```

---

## Structure du projet

```text
PFE_AntiSpoofing_v2/
├── api/                         API FastAPI
├── src/
│   ├── pad_system/              Système PAD final
│   ├── behavior/                Features comportementales, rPPG, landmarks
│   └── deep_learning/           Modèles, datasets, métriques
├── frontend/
│   └── banking_demo/            Interface SecureBank
├── models/                      Modèles externes et MediaPipe
├── experiments/
│   └── 03_final_models/         Modèles finaux retenus
├── data/
│   ├── runtime_uploads/         Vidéos uploadées
│   ├── runtime_results/         Résultats JSON
│   └── runtime_challenges/      Challenges actifs
├── reports/                     Résultats et analyses
├── tests/                       Tests et démonstrations
├── logs/                        Logs API
└── requirements.txt             Dépendances Python
```

---

## Modèles utilisés

### Modèle vidéo principal V6

```text
experiments/03_final_models/video_v6_behavior_pose/
└── mixed_casia_axon_local_msu_gated_hard_balanced_rppg_v3/
    └── seed42/
        └── best_model.pth
```

### Modèle behavior-pose V6

```text
experiments/03_final_models/banking_demo_models/final_models/
└── behavior_pose_v6/
    └── behavior_pose_clf.pkl
```

### Modèle MediaPipe

```text
models/face_landmarker.task
```

---

## Démonstration SecureBank

Le parcours de démonstration suit les étapes suivantes :

1. Ouvrir l’interface **SecureBank**.
2. Cliquer sur **Commencer la vérification**.
3. Démarrer la capture vidéo.
4. Suivre le challenge de vivacité affiché.
5. Attendre l’analyse PAD V6.
6. Consulter la décision finale : `ACCEPT`, `RETRY` ou `REJECT`.
7. Ouvrir le mode administrateur pour consulter :
   - les scores ;
   - le temps d’exécution ;
   - les résultats JSON ;
   - les exports techniques.

---

## Résultats runtime

Chaque analyse génère un fichier JSON dans :

```text
data/runtime_results/
```

Ces fichiers contiennent notamment :

- la décision finale ;
- le score PAD ;
- le statut de liveness ;
- le challenge utilisé ;
- la qualité vidéo ;
- les scores V6 détaillés ;
- le temps d’exécution ;
- le chemin de la vidéo analysée.

---

## Interface utilisateur

L’interface **SecureBank** permet de simuler un parcours bancaire complet :

- page d’accueil explicative ;
- lancement d’une vérification ;
- affichage du challenge actif ;
- capture vidéo via caméra ;
- affichage de la décision finale ;
- mode administrateur ;
- consultation de l’historique ;
- export des résultats.

---

## Technologies principales

| Composant | Technologie |
|---|---|
| Backend API | FastAPI |
| Deep Learning | PyTorch |
| Traitement vidéo | OpenCV |
| Landmarks / liveness | MediaPipe |
| Frontend démo | HTML, CSS, JavaScript |
| Journalisation | JSON |
| Modèle principal | CNN-LSTM-BEHAVIOR multimodal |
| Fusion | Score-level fusion |

---

## Décisions métier

| Décision | Condition générale |
|---|---|
| `ACCEPT` | Liveness validé, score PAD faible, qualité vidéo suffisante |
| `RETRY` | Qualité ou score ambigu, nouvelle capture recommandée |
| `REJECT` | Risque d’attaque ou non-respect du challenge |

---

## Objectif du prototype

Ce prototype vise à démontrer la faisabilité d’un système PAD vidéo intégré dans un parcours bancaire, avec :

- une analyse multimodale ;
- une prise de décision métier ;
- une interface démonstrative claire ;
- une traçabilité des résultats ;
- une architecture organisée et extensible.

---
## Accès HTTPS distant via ngrok

 1. Lancer Docker normalement
     docker compose up -d

  2. Lancer ngrok
     ngrok http 8080

  3. Ouvrir l'URL HTTPS générée par ngrok
     https://livable-squishy-radiance.ngrok-free.dev/

  Cette URL permet :
  → Accès distant au prototype
  → Utilisation de la caméra depuis un téléphone
  → Test réaliste du parcours utilisateur

  Note : ngrok est utilisé pour la démonstration uniquement.
 

---



## Note

Ce projet est un prototype académique et expérimental.  
Une validation supplémentaire serait nécessaire avant une utilisation dans un environnement bancaire réel.
