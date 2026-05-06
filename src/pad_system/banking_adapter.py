def banking_decision(
    score: float,
    profile: str = "video",
) -> str:
    """
    Convertit un score spoof en décision bancaire.

    Convention :
    - score proche de 0 => REAL
    - score proche de 1 => SPOOF

    Version finale adaptée au flux vidéo challenge direct.

    Politique vidéo :
    - score < 0.30          => ACCEPT
    - 0.30 <= score < 0.60  => RETRY
    - score >= 0.60         => REJECT

    Remarque :
    La branche image/CelebA est désactivée dans l'architecture finale.
    Ce fichier ne retourne donc plus VIDEO_CHALLENGE_REQUIRED.
    """

    score = float(score)

    # Profil unique actif dans le système final : vidéo.
    if profile not in ["video", "default", "banking"]:
        profile = "video"

    t_accept = 0.30
    t_reject = 0.60

    if score < t_accept:
        return "ACCEPT"

    if score < t_reject:
        return "RETRY"

    return "REJECT"


def label_from_decision(decision: str) -> str:
    """
    Convertit une décision bancaire en label lisible.
    """

    if decision == "ACCEPT":
        return "REAL"

    if decision == "REJECT":
        return "SPOOF"

    return "UNCERTAIN"