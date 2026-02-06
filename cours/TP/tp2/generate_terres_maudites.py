# -*- coding: utf-8 -*-
"""
Générateur de données synthétiques "Terres Maudites"
Basé sur les règles du README pour le test secret.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_terres_maudites_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Génère des données selon les règles des Terres Maudites.
    
    Formule de survie (d'après le README):
    - Intelligence: 30%
    - Agilité: 20%
    - Chance: 20%
    - Équipement: 15%
    - Force (<70): 10%
    - Expérience: 5%
    - Fatigue: -10%
    - Difficulté: -10%
    - ARROGANCE (Force > 70): -15%
    """
    np.random.seed(seed)
    
    # Générer les features aléatoirement (comme dans le train)
    data = {
        'force': np.random.uniform(0, 100, n_samples),
        'intelligence': np.random.uniform(0, 100, n_samples),
        'agilite': np.random.uniform(0, 100, n_samples),
        'chance': np.random.uniform(0, 100, n_samples),
        'experience': np.random.uniform(0, 20, n_samples),  # Années
        'niveau_quete': np.random.uniform(1, 10, n_samples),
        'equipement': np.random.uniform(0, 100, n_samples),
        'fatigue': np.random.uniform(0, 100, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Normaliser toutes les features entre 0 et 1 pour le calcul
    # (Le README dit que les données de test sont normalisées !)
    df_norm = df.copy()
    df_norm['force'] = df['force'] / 100
    df_norm['intelligence'] = df['intelligence'] / 100
    df_norm['agilite'] = df['agilite'] / 100
    df_norm['chance'] = df['chance'] / 100
    df_norm['experience'] = df['experience'] / 20
    df_norm['niveau_quete'] = (df['niveau_quete'] - 1) / 9
    df_norm['equipement'] = df['equipement'] / 100
    df_norm['fatigue'] = df['fatigue'] / 100
    
    # Calculer le score de survie selon les règles Terres Maudites
    score = (
        0.30 * df_norm['intelligence'] +
        0.20 * df_norm['agilite'] +
        0.20 * df_norm['chance'] +
        0.15 * df_norm['equipement'] +
        0.10 * df_norm['force'] +  # Force compte moins
        0.05 * df_norm['experience'] -
        0.10 * df_norm['fatigue'] -
        0.10 * df_norm['niveau_quete']
    )
    
    # Piège de l'Arrogance : Force > 70 = malus !
    arrogance_mask = df['force'] > 70
    score[arrogance_mask] -= 0.15
    
    # Ajouter du bruit
    score += np.random.normal(0, 0.05, n_samples)
    
    # Convertir en label binaire (seuil à 0.5)
    df['survie'] = (score > 0.35).astype(int)
    
    print(f"Données générées: {n_samples} échantillons")
    print(f"Survie: {df['survie'].mean():.1%}")
    print(f"Force > 70 (Arrogants): {arrogance_mask.sum()} ({arrogance_mask.mean():.1%})")
    
    return df


def evaluate_on_terres_maudites(model, df: pd.DataFrame, normalize: bool = True):
    """
    Évalue un modèle sur les données Terres Maudites synthétiques.
    """
    import torch
    
    features = df.drop('survie', axis=1).values
    labels = df['survie'].values
    
    if normalize:
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features = (features - mean) / std
    
    X = torch.tensor(features, dtype=torch.float32)
    y = labels
    
    model.eval()
    with torch.no_grad():
        predictions = model.predict(X).squeeze().numpy()
    
    accuracy = (predictions == y).mean()
    return accuracy


if __name__ == "__main__":
    # Générer les données
    df = generate_terres_maudites_data(500)
    
    # Sauvegarder
    output_path = Path(__file__).parent / "data" / "synthetic_terres_maudites.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSauvegardé: {output_path}")
    
    # Tester le modèle actuel
    try:
        import torch
        from baseline_model import GuildOracle
        
        checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
        if checkpoint_path.exists():
            model = torch.load(checkpoint_path)
            acc = evaluate_on_terres_maudites(model, df, normalize=True)
            print(f"\nAccuracy sur Terres Maudites synthétiques: {acc:.2%}")
    except Exception as e:
        print(f"Erreur lors du test: {e}")
