def quality_level(video_quality_score: float) -> str:
    q = float(video_quality_score)

    if q >= 0.70:
        return "GOOD"

    if q >= 0.45:
        return "MEDIUM"

    return "LOW"


def banking_decision(
    score: float,
    profile: str = "video",
    video_quality_score: float = 0.70,
) -> str:
    """
    Politique bancaire finale quality-aware.

    Convention :
    - score proche de 0 => REAL
    - score proche de 1 => SPOOF
    """

    score = float(score)
    level = quality_level(video_quality_score)

    if level == "GOOD":
        t_accept = 0.30
        t_reject = 0.80

    elif level == "MEDIUM":
        t_accept = 0.25
        t_reject = 0.75

    else:
        t_accept = 0.20
        t_reject = 0.80

    if score < t_accept:
        return "ACCEPT"

    if score < t_reject:
        return "RETRY"

    return "REJECT"


def label_from_decision(decision: str) -> str:
    if decision == "ACCEPT":
        return "REAL"

    if decision == "REJECT":
        return "SPOOF"

    return "UNCERTAIN"


def next_action_from_decision(decision: str) -> str:
    if decision == "ACCEPT":
        return "NONE"

    if decision == "REJECT":
        return "BLOCK_OR_MANUAL_REVIEW"

    return "RETRY_VIDEO_CAPTURE"


def message_from_decision(decision: str) -> str:
    if decision == "ACCEPT":
        return "Analyse video terminee. Video acceptee."

    if decision == "REJECT":
        return "Vidéo rejetee. Suspicion d'attaque de presentation."

    return "Analyse video ambiguë. Nouvelle capture recommandée."
