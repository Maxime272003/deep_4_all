"""
Phase 3 : Génération du Dataset via API Teacher
Génère les réponses de raisonnement pour CommonsenseQA
"""

import json
import time
import requests
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
from datasets import load_dataset

from config import (
    INFOMANIAK_API_URL,
    INFOMANIAK_API_KEY,
    TEACHER_MODEL,
    SYSTEM_PROMPT,
    MAX_TOKENS,
    TEMPERATURE_STAGE1,
    TEMPERATURE_STAGE2,
    NUM_SAMPLES_STAGE1,
    NUM_SAMPLES_STAGE2,
    API_DELAY_SECONDS,
    DATA_DIR,
    LOGS_DIR
)


@dataclass
class GeneratedExample:
    """Structure d'un exemple généré"""
    instruction: str
    response: str
    answer_key: str
    temperature: float
    question_id: Optional[str] = None


class DatasetGenerator:
    """Générateur de dataset via API Teacher"""
    
    def __init__(self):
        self.api_url = INFOMANIAK_API_URL
        self.api_key = INFOMANIAK_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.log_file = LOGS_DIR / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log(self, message: str):
        """Logger les événements"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
    
    @staticmethod
    def format_csqa_prompt(example: Dict) -> str:
        """
        Formate une question CommonsenseQA en prompt pour le Teacher
        
        Args:
            example: Un exemple du dataset CommonsenseQA avec 'question' et 'choices'
            
        Returns:
            Prompt formaté pour le modèle Teacher
        """
        question = example["question"]
        choices = example["choices"]
        
        # Construire les options
        options_text = "\n".join([
            f"{label}. {text}" 
            for label, text in zip(choices["label"], choices["text"])
        ])
        
        prompt = f"""Question: {question}

Options:
{options_text}

Please analyze this question step by step and select the correct answer."""
        
        return prompt
    
    def generate_teacher_response(
        self, 
        prompt: str, 
        temperature: float = 0.3,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Génère une réponse du Teacher
        
        Args:
            prompt: Le prompt utilisateur
            temperature: Température de génération
            max_retries: Nombre de tentatives en cas d'erreur
            
        Returns:
            Dict avec response et temperature, ou None si échec
        """
        payload = {
            "model": TEACHER_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                return {
                    "response": content,
                    "temperature": temperature
                }
                
            except Exception as e:
                self.log(f"Erreur API (tentative {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # Backoff exponentiel
                    
        return None
    
    def filter_response_quality(self, response: str, answer_key: str) -> bool:
        """
        Filtre les réponses de mauvaise qualité
        
        Args:
            response: La réponse générée
            answer_key: La bonne réponse attendue
            
        Returns:
            True si la réponse est de bonne qualité
        """
        # Vérifier que la réponse contient du raisonnement
        has_reasoning = "<reasoning>" in response.lower() or len(response) > 200
        
        # Vérifier que la réponse n'est pas vide
        not_empty = len(response.strip()) > 50
        
        # Vérifier que la réponse contient une conclusion
        has_answer = any(
            pattern in response.upper() 
            for pattern in ["FINAL ANSWER", "THE ANSWER IS", "ANSWER:", f"({answer_key.upper()})"]
        )
        
        return has_reasoning and not_empty
    
    def generate_stage_dataset(
        self,
        temperature: float,
        num_samples: int,
        start_idx: int = 0,
        stage_name: str = "stage1"
    ) -> List[GeneratedExample]:
        """
        Génère un dataset pour un stage spécifique
        
        Args:
            temperature: Température de génération
            num_samples: Nombre d'exemples à générer
            start_idx: Index de départ dans le dataset source
            stage_name: Nom du stage pour les logs
            
        Returns:
            Liste d'exemples générés
        """
        self.log(f"=== Génération {stage_name} (τ={temperature}) ===")
        self.log(f"Chargement de CommonsenseQA...")
        
        # Charger CommonsenseQA
        csqa = load_dataset("commonsense_qa", split="train")
        
        generated_data = []
        failed_count = 0
        
        # Sélectionner les exemples
        end_idx = min(start_idx + num_samples, len(csqa))
        examples = csqa.select(range(start_idx, end_idx))
        
        self.log(f"Génération de {len(examples)} exemples...")
        
        for i, example in enumerate(tqdm(examples, desc=f"Génération {stage_name}")):
            prompt = self.format_csqa_prompt(example)
            result = self.generate_teacher_response(prompt, temperature)
            
            if result and self.filter_response_quality(result["response"], example["answerKey"]):
                gen_example = GeneratedExample(
                    instruction=prompt,
                    response=result["response"],
                    answer_key=example["answerKey"],
                    temperature=temperature,
                    question_id=example.get("id", f"q_{start_idx + i}")
                )
                generated_data.append(gen_example)
            else:
                failed_count += 1
                self.log(f"Échec pour l'exemple {start_idx + i}")
            
            # Rate limiting
            time.sleep(API_DELAY_SECONDS)
        
        self.log(f"Terminé: {len(generated_data)} succès, {failed_count} échecs")
        return generated_data
    
    def save_raw_dataset(self, data: List[GeneratedExample], filename: str):
        """Sauvegarde le dataset brut avec les logprobs"""
        filepath = DATA_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([asdict(ex) for ex in data], f, ensure_ascii=False, indent=2)
        self.log(f"Dataset sauvegardé: {filepath}")
    
    def convert_to_sharegpt(self, data: List[GeneratedExample], filename: str):
        """
        Convertit au format ShareGPT pour Llama-Factory
        
        Args:
            data: Liste d'exemples générés
            filename: Nom du fichier de sortie
        """
        sharegpt_data = []
        for ex in data:
            sharegpt_data.append({
                "conversations": [
                    {"from": "human", "value": ex.instruction},
                    {"from": "gpt", "value": ex.response}
                ]
            })
        
        filepath = DATA_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
        self.log(f"Dataset ShareGPT sauvegardé: {filepath}")
    
    def run(self):
        """Exécute la génération complète des deux stages"""
        self.log("=" * 50)
        self.log("DÉMARRAGE DE LA GÉNÉRATION DU DATASET")
        self.log("=" * 50)
        
        # Stage 1 : Basse température
        stage1_data = self.generate_stage_dataset(
            temperature=TEMPERATURE_STAGE1,
            num_samples=NUM_SAMPLES_STAGE1,
            start_idx=0,
            stage_name="stage1"
        )
        
        # Sauvegarder Stage 1
        self.save_raw_dataset(stage1_data, "csqa_stage1_raw.json")
        self.convert_to_sharegpt(stage1_data, "csqa_stage1_sharegpt.json")
        
        # Stage 2 : Haute température
        stage2_data = self.generate_stage_dataset(
            temperature=TEMPERATURE_STAGE2,
            num_samples=NUM_SAMPLES_STAGE2,
            start_idx=NUM_SAMPLES_STAGE1,  # Commencer après Stage 1
            stage_name="stage2"
        )
        
        # Sauvegarder Stage 2
        self.save_raw_dataset(stage2_data, "csqa_stage2_raw.json")
        self.convert_to_sharegpt(stage2_data, "csqa_stage2_sharegpt.json")
        
        self.log("=" * 50)
        self.log("GÉNÉRATION TERMINÉE")
        self.log(f"Stage 1: {len(stage1_data)} exemples")
        self.log(f"Stage 2: {len(stage2_data)} exemples")
        self.log("=" * 50)
        
        return stage1_data, stage2_data


if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.run()
