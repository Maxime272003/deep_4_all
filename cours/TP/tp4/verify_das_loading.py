import sys
import os

# Add local dir to path
sys.path.append(os.getcwd())

from das_filtering import DASFilter

def verify():
    print("=== Vérification du chargement DASFilter ===")
    try:
        # Initialisation (charge le modèle)
        print("Initialisation de DASFilter...")
        # Note: ceci va prendre du temps (téléchargement/chargement modèle)
        das = DASFilter(load_model=True)
        print("✅ DASFilter initialisé avec succès!")
        
        if das.model:
            print(f"Device du modèle: {das.model.device}")
            print(f"Config du modèle: {das.model.config}")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
