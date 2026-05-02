def banking_decision(score: float) -> str:
    """
    Décision bancaire à 3 niveaux.

    score = probabilité que l'entrée soit une attaque spoof.
    """

    T_ACCEPT = 0.30
    T_REJECT = 0.60

    if score < T_ACCEPT:
        return "ACCEPT"

    elif score < T_REJECT:
        return "RETRY"

    else:
        return "REJECT"


def label_from_decision(decision: str) -> str:
    """
    Convertit la décision bancaire en label lisible.
    """

    if decision == "ACCEPT":
        return "REAL"

    elif decision == "REJECT":
        return "SPOOF"

    else:
        return "UNCERTAIN"