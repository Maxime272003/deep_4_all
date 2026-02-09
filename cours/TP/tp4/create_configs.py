"""
Phase 5 : Création des fichiers de configuration pour Llama-Factory
"""

import json
import yaml
from config import *


def create_dataset_info():
    """Crée dataset_info.json pour Llama-Factory"""
    dataset_info = {
        "csqa_stage1": {
            "file_name": str(DATA_DIR / "csqa_stage1_filtered.json"),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"}
        },
        "csqa_stage2": {
            "file_name": str(DATA_DIR / "csqa_stage2_filtered.json"),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"}
        }
    }
    
    output_path = CONFIGS_DIR / "dataset_info.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Créé: {output_path}")


def create_stage1_config():
    """Config YAML pour Stage 1"""
    config = {
        "model_name_or_path": STUDENT_MODEL_ID,
        "stage": "sft", "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_target": "all",
        "dataset": "csqa_stage1", "dataset_dir": str(CONFIGS_DIR),
        "template": "qwen", "cutoff_len": CUTOFF_LEN,
        "output_dir": str(CHECKPOINTS_DIR / "csqa_stage1"),
        "logging_steps": 10, "save_steps": 100,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE, "num_train_epochs": NUM_EPOCHS,
        "bf16": True, "gradient_checkpointing": True, "report_to": "none"
    }
    
    with open(CONFIGS_DIR / "stage1_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Créé: {CONFIGS_DIR / 'stage1_config.yaml'}")


def create_stage2_config():
    """Config YAML pour Stage 2 (charge adapter Stage 1)"""
    config = {
        "model_name_or_path": STUDENT_MODEL_ID,
        "adapter_name_or_path": str(CHECKPOINTS_DIR / "csqa_stage1"),
        "stage": "sft", "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_target": "all",
        "dataset": "csqa_stage2", "dataset_dir": str(CONFIGS_DIR),
        "template": "qwen", "cutoff_len": CUTOFF_LEN,
        "output_dir": str(CHECKPOINTS_DIR / "csqa_stage2"),
        "logging_steps": 10, "save_steps": 100,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE * 0.5, "num_train_epochs": NUM_EPOCHS,
        "bf16": True, "gradient_checkpointing": True, "report_to": "none"
    }
    
    with open(CONFIGS_DIR / "stage2_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Créé: {CONFIGS_DIR / 'stage2_config.yaml'}")


if __name__ == "__main__":
    print("=== CRÉATION DES CONFIGURATIONS ===")
    create_dataset_info()
    create_stage1_config()
    create_stage2_config()
    print("\nPour lancer l'entraînement:")
    print(f"  llamafactory-cli train {CONFIGS_DIR / 'stage1_config.yaml'}")
    print(f"  llamafactory-cli train {CONFIGS_DIR / 'stage2_config.yaml'}")
