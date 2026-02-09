# TP4 DASD - CommonsenseQA

## Description
Distillation de modèles de raisonnement avec DASD (Distribution-Aligned Sequence Distillation) sur le dataset CommonsenseQA.

## Structure
```
tp4/
├── config.py              # Configuration centrale
├── generate_dataset.py    # Phase 3: Génération via API
├── das_filtering.py       # Phase 4: Filtrage DAS
├── create_configs.py      # Phase 5: Configs Llama-Factory
├── evaluate.py            # Phase 9: Évaluation
├── main.py                # Script principal
├── data/                  # Datasets générés
├── configs/               # Fichiers YAML
├── checkpoints/           # Adapters LoRA
└── logs/                  # Logs et graphiques
```

## Installation
```bash
pip install llama-factory transformers datasets openai torch bitsandbytes accelerate nltk peft matplotlib tqdm pyyaml
```

## Configuration
1. Définir la clé API dans `config.py` ou via variable d'environnement:
```bash
export INFOMANIAK_API_KEY="votre_cle"
```

## Exécution

### Pipeline complet
```bash
python main.py all
```

### Étape par étape
```bash
# 1. Génération du dataset via API
python main.py generate

# 2. Filtrage DAS
python main.py das

# 3. Création des configs
python main.py configs

# 4. Entraînement Stage 1
python main.py train --stage 1

# 5. Entraînement Stage 2
python main.py train --stage 2

# 6. Évaluation
python main.py evaluate
```

## Outputs
- `data/csqa_stage1_filtered.json` - Dataset Stage 1 filtré
- `data/csqa_stage2_filtered.json` - Dataset Stage 2 filtré
- `checkpoints/csqa_stage1/` - Adapter LoRA Stage 1
- `checkpoints/csqa_stage2/` - Adapter LoRA Stage 2
- `logs/eval_*.json` - Résultats d'évaluation
- `logs/das_*.png` - Graphiques de distribution DAS
