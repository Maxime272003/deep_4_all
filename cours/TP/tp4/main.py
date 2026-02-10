"""
Script principal - Orchestre toutes les phases du TP4
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="TP4 DASD - Pipeline complet")
    parser.add_argument("phase", choices=[
        "generate", "das", "configs", "train", "evaluate", "all"
    ], help="Phase à exécuter")
    parser.add_argument("--stage", choices=["1", "2"], help="Stage pour train")
    
    args = parser.parse_args()
    
    if args.phase == "generate":
        print("=== PHASE 3: Génération du Dataset ===")
        from generate_dataset import DatasetGenerator
        generator = DatasetGenerator()
        generator.run()
        
    elif args.phase == "das":
        print("=== PHASE 4: Filtrage DAS ===")
        from das_filtering import run_das_filtering
        run_das_filtering()
        
    elif args.phase == "configs":
        print("=== PHASE 5: Création des Configs ===")
        from create_configs import create_dataset_info, create_stage1_config, create_stage2_config, create_inference_config
        create_dataset_info()
        create_stage1_config()
        create_stage2_config()
        create_inference_config()
        
    elif args.phase == "train":
        print("=== PHASE 6: Entraînement ===")
        from config import CONFIGS_DIR
        import subprocess
        
        if args.stage == "1":
            config_file = CONFIGS_DIR / "stage1_config.yaml"
        elif args.stage == "2":
            config_file = CONFIGS_DIR / "stage2_config.yaml"
        else:
            print("Spécifiez --stage 1 ou --stage 2")
            sys.exit(1)
        
        cmd = f"llamafactory-cli train {config_file}"
        print(f"Exécution: {cmd}")
        subprocess.run(cmd, shell=True)
        
    elif args.phase == "evaluate":
        print("=== PHASE 9: Évaluation ===")
        from evaluate import run_evaluation
        run_evaluation()
        
    elif args.phase == "all":
        print("=== EXÉCUTION COMPLÈTE ===")
        # Génération
        from generate_dataset import DatasetGenerator
        generator = DatasetGenerator()
        generator.run()
        
        # DAS
        from das_filtering import run_das_filtering
        run_das_filtering()
        
        # Configs
        from create_configs import create_dataset_info, create_stage1_config, create_stage2_config
        create_dataset_info()
        create_stage1_config()
        create_stage2_config()
        
        print("\n>>> Configs créées. Lancez manuellement:")
        print("    python main.py train --stage 1")
        print("    python main.py train --stage 2")
        print("    python main.py evaluate")


if __name__ == "__main__":
    main()
