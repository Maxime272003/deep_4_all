"""
Configuration centrale pour le TP4 - DASD avec CommonsenseQA
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# === CHARGEMENT DU FICHIER .env ===
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# === CHEMINS ===
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CONFIGS_DIR = BASE_DIR / "configs"
LOGS_DIR = BASE_DIR / "logs"

# Créer les dossiers s'ils n'existent pas
for dir_path in [DATA_DIR, CHECKPOINTS_DIR, CONFIGS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# === API INFOMANIAK (TEACHER) ===
INFOMANIAK_API_URL = os.getenv("INFOMANIAK_API_URL", "https://api.infomaniak.com/2/ai/48/openai/v1/chat/completions")
INFOMANIAK_API_KEY = os.getenv("INFOMANIAK_API_KEY")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "openai/gpt-oss-120b")

# === MODÈLE ÉTUDIANT ===
STUDENT_MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

# === PARAMÈTRES DE GÉNÉRATION ===
MAX_TOKENS = 2000
TEMPERATURE_STAGE1 = 0.3  
TEMPERATURE_STAGE2 = 0.9  

# === SYSTEM PROMPT POUR LE RAISONNEMENT ===
SYSTEM_PROMPT = """You are a helpful assistant that reasons step by step. 
Always structure your reasoning inside <reasoning>...</reasoning> tags before giving your final answer. 
Be thorough in your reasoning process.
For multiple choice questions, analyze each option carefully before selecting your answer.
After your reasoning, clearly state your final answer as: "Final Answer: [LETTER]" where [LETTER] is A, B, C, D, or E."""

# === PARAMÈTRES DAS ===
DAS_P_TEACHER_THRESHOLD = 0.6  
DAS_DIVERGENCE_THRESHOLD = 0.2  
DAS_MIN_TEACHER_SENTENCES_RATIO = 0.3  

# === PARAMÈTRES D'ENTRAÎNEMENT ===
LORA_RANK = 16
LORA_ALPHA = 32
CUTOFF_LEN = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# === DATASET ===
NUM_SAMPLES_STAGE1 = 550  
NUM_SAMPLES_STAGE2 = 550  
API_DELAY_SECONDS = 0.5  
