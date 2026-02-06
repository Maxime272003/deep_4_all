# -*- coding: utf-8 -*-
"""
Affinement de la formule Terres Maudites.
Teste plusieurs hypotheses et trouve la meilleure.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from baseline_model import GuildOracle, count_parameters


def generate_tm_data(formula_id: str, n_samples: int = 500, seed: int = 42):
    """Genere des donnees avec differentes formules."""
    np.random.seed(seed)
    
    data = {
        'force': np.random.uniform(0, 100, n_samples),
        'intelligence': np.random.uniform(0, 100, n_samples),
        'agilite': np.random.uniform(0, 100, n_samples),
        'chance': np.random.uniform(0, 100, n_samples),
        'experience': np.random.uniform(0, 20, n_samples),
        'niveau_quete': np.random.uniform(1, 10, n_samples),
        'equipement': np.random.uniform(0, 100, n_samples),
        'fatigue': np.random.uniform(0, 100, n_samples),
    }
    df = pd.DataFrame(data)
    
    # Normaliser
    df_n = df.copy()
    df_n['force'] = df['force'] / 100
    df_n['intelligence'] = df['intelligence'] / 100
    df_n['agilite'] = df['agilite'] / 100
    df_n['chance'] = df['chance'] / 100
    df_n['experience'] = df['experience'] / 20
    df_n['niveau_quete'] = (df['niveau_quete'] - 1) / 9
    df_n['equipement'] = df['equipement'] / 100
    df_n['fatigue'] = df['fatigue'] / 100
    
    # Differentes formules a tester
    if formula_id == "v1_base":
        # Formule originale du README
        score = (0.30*df_n['intelligence'] + 0.20*df_n['agilite'] + 0.20*df_n['chance'] +
                 0.15*df_n['equipement'] + 0.10*df_n['force'] + 0.05*df_n['experience'] -
                 0.10*df_n['fatigue'] - 0.10*df_n['niveau_quete'])
        score[df['force'] > 70] -= 0.15
        
    elif formula_id == "v2_intel_boost":
        # Intelligence encore plus importante
        score = (0.40*df_n['intelligence'] + 0.20*df_n['agilite'] + 0.15*df_n['chance'] +
                 0.10*df_n['equipement'] + 0.05*df_n['force'] + 0.05*df_n['experience'] -
                 0.10*df_n['fatigue'] - 0.10*df_n['niveau_quete'])
        score[df['force'] > 70] -= 0.20
        
    elif formula_id == "v3_force_penalty":
        # Penalite force plus severe
        score = (0.30*df_n['intelligence'] + 0.25*df_n['agilite'] + 0.20*df_n['chance'] +
                 0.10*df_n['equipement'] + 0.00*df_n['force'] + 0.05*df_n['experience'] -
                 0.10*df_n['fatigue'] - 0.10*df_n['niveau_quete'])
        score[df['force'] > 60] -= 0.20  # Seuil plus bas!
        
    elif formula_id == "v4_luck_matters":
        # La chance compte beaucoup plus
        score = (0.25*df_n['intelligence'] + 0.15*df_n['agilite'] + 0.30*df_n['chance'] +
                 0.10*df_n['equipement'] + 0.05*df_n['force'] + 0.05*df_n['experience'] -
                 0.10*df_n['fatigue'] - 0.10*df_n['niveau_quete'])
        score[df['force'] > 70] -= 0.15
        
    elif formula_id == "v5_inverse":
        # Inversion totale des poids (force devient negatif)
        score = (0.35*df_n['intelligence'] + 0.25*df_n['agilite'] + 0.25*df_n['chance'] +
                 0.10*df_n['equipement'] - 0.10*df_n['force'] + 0.05*df_n['experience'] -
                 0.15*df_n['fatigue'] - 0.15*df_n['niveau_quete'])
        
    elif formula_id == "v6_equipment_key":
        # L'equipement est crucial dans les TM aussi
        score = (0.25*df_n['intelligence'] + 0.20*df_n['agilite'] + 0.15*df_n['chance'] +
                 0.25*df_n['equipement'] + 0.00*df_n['force'] + 0.05*df_n['experience'] -
                 0.10*df_n['fatigue'] - 0.10*df_n['niveau_quete'])
        score[df['force'] > 70] -= 0.15
        
    else:
        raise ValueError(f"Unknown formula: {formula_id}")
    
    score += np.random.normal(0, 0.05, n_samples)
    df['survie'] = (score > 0.35).astype(int)
    
    return df


def test_formula(formula_id, train_df, val_df, config):
    """Entraine sur train, teste sur les donnees TM synthetiques."""
    tm_df = generate_tm_data(formula_id)
    
    # Prepare data
    X_train = torch.tensor(train_df.drop('survie', axis=1).values, dtype=torch.float32)
    y_train = torch.tensor(train_df['survie'].values, dtype=torch.float32)
    
    # Normaliser train
    train_mean = X_train.mean(dim=0)
    train_std = X_train.std(dim=0) + 1e-8
    X_train = (X_train - train_mean) / train_std
    
    X_val = torch.tensor(val_df.drop('survie', axis=1).values, dtype=torch.float32)
    y_val = torch.tensor(val_df['survie'].values, dtype=torch.float32)
    X_val = (X_val - train_mean) / train_std
    
    # TM avec SES PROPRES stats
    X_tm = torch.tensor(tm_df.drop('survie', axis=1).values, dtype=torch.float32)
    y_tm = torch.tensor(tm_df['survie'].values, dtype=torch.float32)
    tm_mean = X_tm.mean(dim=0)
    tm_std = X_tm.std(dim=0) + 1e-8
    X_tm = (X_tm - tm_mean) / tm_std
    
    # Train
    model = GuildOracle(input_dim=8, **config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    for _ in range(50):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train).squeeze(), y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_preds = (torch.sigmoid(model(X_val).squeeze()) > 0.5).float()
        val_acc = (val_preds == y_val).float().mean().item()
        
        tm_preds = (torch.sigmoid(model(X_tm).squeeze()) > 0.5).float()
        tm_acc = (tm_preds == y_tm).float().mean().item()
    
    return val_acc, tm_acc, model


def main():
    data_dir = Path(__file__).parent / "data"
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    
    formulas = ["v1_base", "v2_intel_boost", "v3_force_penalty", 
                "v4_luck_matters", "v5_inverse", "v6_equipment_key"]
    
    configs = [
        {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
        {"hidden_dim": 6, "num_layers": 1, "dropout": 0.0},
        {"hidden_dim": 8, "num_layers": 1, "dropout": 0.2},
    ]
    
    print("=" * 70)
    print("TEST DE DIFFERENTES FORMULES TERRES MAUDITES")
    print("=" * 70)
    
    best_results = []
    
    for formula in formulas:
        print(f"\n--- Formule: {formula} ---")
        for config in configs:
            val_acc, tm_acc, model = test_formula(formula, train_df, val_df, config)
            gap = val_acc - tm_acc
            print(f"  Config {config['hidden_dim']}: Val={val_acc:.2%}, TM={tm_acc:.2%}, Gap={gap:+.2%}")
            best_results.append({
                'formula': formula,
                'config': config,
                'val_acc': val_acc,
                'tm_acc': tm_acc,
                'gap': gap
            })
    
    # Trouver celle avec le plus petit gap (meilleure generalisation)
    best_results.sort(key=lambda x: abs(x['gap']))
    
    print("\n" + "=" * 70)
    print("MEILLEURS RESULTATS (par gap le plus petit)")
    print("=" * 70)
    for r in best_results[:5]:
        print(f"\n{r['formula']}, hidden={r['config']['hidden_dim']}")
        print(f"  Val: {r['val_acc']:.2%}, TM: {r['tm_acc']:.2%}, Gap: {r['gap']:+.2%}")


if __name__ == "__main__":
    main()
