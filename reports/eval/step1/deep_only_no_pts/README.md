CNN_LSTM   → répond à : "ton modèle deep seul, il fait quoi ?"

deep_only  → répond à : "dans ton pipeline complet,
              AVANT d'ajouter behavioral et PTS, ça donne quoi ?"
              
Ce sont deux sessions d'entraînement différentes. CNN_LSTM est entraîné en step 1 isolément, deep_only est entraîné dans le pipeline complet. La différence de performance est normale et attendue — ce qui compte c'est la comparaison interne à chaque étape.
