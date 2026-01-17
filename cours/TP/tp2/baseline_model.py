"""
Modèle Baseline : Oracle de la Guilde (Non Optimal)

Ce modèle est VOLONTAIREMENT non optimal.
Les étudiants doivent identifier et corriger les problèmes !
"""

import torch
import torch.nn as nn


class GuildOracle(nn.Module):
    """
    Modèle baseline pour prédire la survie des aventuriers.

    Architecture : MLP profond (trop profond !)
    """

    def __init__(self, input_dim: int = 12, hidden_dim: int = 256, num_layers: int = 5):
        """
        Args:
            input_dim: Nombre de features (8 stats + 4 bruit = 12)
            hidden_dim: Dimension des couches cachées
            num_layers: Nombre de couches cachées
        """
        super().__init__()

        # Construction des couches
        layers = []

        # Première couche
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Couches cachées (beaucoup trop !)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Couche de sortie
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch_size, input_dim)

        Returns:
            Logits de shape (batch_size, 1)
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités de survie."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les prédictions binaires."""
        proba = self.predict_proba(x)
        return (proba > 0.5).float()


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Compte le nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_dim: int = 12):
    """Affiche un résumé du modèle."""
    print("=" * 50)
    print("Résumé du modèle")
    print("=" * 50)
    print(model)
    print("-" * 50)
    print(f"Nombre de paramètres : {count_parameters(model):,}")
    print("=" * 50)


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    # Test du modèle baseline
    print("Test du modèle GuildOracle (baseline)")

    model = GuildOracle(input_dim=12)
    model_summary(model)

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 12)

    logits = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    proba = model.predict_proba(x)
    print(f"Probabilités (min, max): ({proba.min():.3f}, {proba.max():.3f})")

    pred = model.predict(x)
    print(f"Prédictions: {pred.sum().item():.0f} survies sur {batch_size}")

    # Avertissement sur l'overfitting
    print("\n" + "!" * 50)
    print("ATTENTION: Ce modèle a beaucoup de paramètres !")
    print("Avec ~800 exemples d'entraînement, il va probablement")
    print("sur-apprendre (overfitting). À améliorer !")
    print("!" * 50)
