# -*- coding: utf-8 -*-
"""
Script pour entrainer et evaluer sur les donnees synthetiques Terres Maudites.
"""

import torch
import pandas as pd
from pathlib import Path
from baseline_model import GuildOracle, count_parameters

def main():
    # Charger les donnees
    data_dir = Path(__file__).parent / "data"
    
    # Donnees d'entrainement originales
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    
    # Donnees synthetiques Terres Maudites
    tm_df = pd.read_csv(data_dir / "synthetic_terres_maudites.csv")
    
    # Preparer les features
    def prepare_data(df, normalize=True, ref_df=None):
        labels = df['survie'].values
        features = df.drop('survie', axis=1).values
        
        if normalize:
            if ref_df is not None:
                ref_features = ref_df.drop('survie', axis=1).values
                mean = ref_features.mean(axis=0)
                std = ref_features.std(axis=0) + 1e-8
            else:
                mean = features.mean(axis=0)
                std = features.std(axis=0) + 1e-8
            features = (features - mean) / std
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
    
    # Configs a tester
    configs = [
        {"hidden_dim": 6, "num_layers": 1, "dropout": 0.0},  # Nano simple
        {"hidden_dim": 10, "num_layers": 1, "dropout": 0.3},  # ~100 params
        {"hidden_dim": 16, "num_layers": 1, "dropout": 0.3},  # ~180 params
        {"hidden_dim": 8, "num_layers": 2, "dropout": 0.3},  # 2 couches
    ]
    
    print("=" * 70)
    print("TEST SUR DONNEES SYNTHETIQUES TERRES MAUDITES")
    print("=" * 70)
    
    for config in configs:
        model = GuildOracle(input_dim=8, **config)
        n_params = count_parameters(model)
        
        # Entrainer rapidement
        X_train, y_train = prepare_data(train_df, normalize=True)
        X_val, y_val = prepare_data(val_df, normalize=True, ref_df=train_df)
        X_tm, y_tm = prepare_data(tm_df, normalize=True)  # Normalise avec ses propres stats!
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Entrainement rapide (30 epochs)
        model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            outputs = model(X_train).squeeze()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Validation originale
            val_preds = (torch.sigmoid(model(X_val).squeeze()) > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()
            
            # Terres Maudites synthetiques
            tm_preds = (torch.sigmoid(model(X_tm).squeeze()) > 0.5).float()
            tm_acc = (tm_preds == y_tm).float().mean().item()
        
        gap = val_acc - tm_acc
        print(f"\nConfig: {config}")
        print(f"  Params: {n_params}")
        print(f"  Val Acc: {val_acc:.2%}")
        print(f"  Terres Maudites Acc: {tm_acc:.2%}")
        print(f"  Gap: {gap:+.2%}")

if __name__ == "__main__":
    main()
