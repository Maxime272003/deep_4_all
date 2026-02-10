"""
Phase 9 : Évaluation du modèle distillé
Compare les performances avant/après distillation
"""

import json
import re
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

from config import (
    STUDENT_MODEL_ID,
    CHECKPOINTS_DIR,
    DATA_DIR,
    LOGS_DIR
)


@dataclass
class EvaluationResult:
    """Résultat d'évaluation pour un exemple"""
    question_id: str
    question: str
    expected_answer: str
    predicted_answer: str
    correct: bool
    full_response: str


class ModelEvaluator:
    """Évaluateur de modèle sur CommonsenseQA"""
    
    def __init__(self, adapter_path: Optional[str] = None):
        """
        Initialise l'évaluateur
        
        Args:
            adapter_path: Chemin vers l'adapter LoRA (None = modèle de base)
        """
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle avec ou sans adapter"""
        print(f"Chargement du modèle: {STUDENT_MODEL_ID}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            STUDENT_MODEL_ID,
            trust_remote_code=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        if self.adapter_path:
            print(f"Chargement de l'adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            self.model = base_model
        
        self.model.eval()
        print("Modèle chargé!")
    
    @staticmethod
    def format_prompt(example: Dict) -> str:
        """Formate un exemple CommonsenseQA"""
        question = example["question"]
        choices = example["choices"]
        
        options_text = "\n".join([
            f"{label}. {text}" 
            for label, text in zip(choices["label"], choices["text"])
        ])
        
        return f"""Question: {question}

Options:
{options_text}

Please analyze this question step by step and select the correct answer."""
    
    def generate_response(
        self, 
        prompt: str, 
        max_new_tokens: int = 500
    ) -> str:
        """Génère une réponse pour un prompt donné"""
        messages = [{"role": "user", "content": prompt}]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response
    
    @staticmethod
    def extract_answer(response: str) -> Optional[str]:
        """
        Extrait la lettre de réponse d'une réponse générée
        
        Args:
            response: La réponse complète du modèle
            
        Returns:
            La lettre (A-E) ou None si non trouvée
        """
        patterns = [
            r"Final Answer:\s*([A-E])",
            r"The answer is\s*([A-E])",
            r"Answer:\s*([A-E])",
            r"\(([A-E])\)\s*$",
            r"option\s+([A-E])",
            r"choice\s+([A-E])",
            r"^([A-E])\.",
            r"select\s+([A-E])",
        ]
        
        response_upper = response.upper()
        
        for pattern in patterns:
            match = re.search(pattern, response_upper, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        matches = re.findall(r'\b([A-E])\b', response_upper)
        if matches:
            return matches[-1]
        
        return None
    
    def evaluate_dataset(
        self,
        split: str = "validation",
        num_samples: int = 100,
        start_idx: int = 0
    ) -> Tuple[float, List[EvaluationResult]]:
        """
        Évalue le modèle sur un split de CommonsenseQA
        
        Args:
            split: Split à utiliser ("validation" ou "test")
            num_samples: Nombre d'exemples à évaluer
            start_idx: Index de départ
            
        Returns:
            Tuple (accuracy, liste des résultats)
        """
        print(f"Chargement de CommonsenseQA ({split})...")
        dataset = load_dataset("commonsense_qa", split=split)
        
        end_idx = min(start_idx + num_samples, len(dataset))
        examples = dataset.select(range(start_idx, end_idx))
        
        results = []
        correct = 0
        
        print(f"Évaluation de {len(examples)} exemples...")
        for i, example in enumerate(tqdm(examples, desc="Évaluation")):
            prompt = self.format_prompt(example)
            response = self.generate_response(prompt)
            
            predicted = self.extract_answer(response)
            expected = example["answerKey"]
            is_correct = predicted == expected
            
            if is_correct:
                correct += 1
            
            results.append(EvaluationResult(
                question_id=example.get("id", f"q_{start_idx + i}"),
                question=example["question"],
                expected_answer=expected,
                predicted_answer=predicted or "NONE",
                correct=is_correct,
                full_response=response
            ))
        
        accuracy = correct / len(examples)
        return accuracy, results
    
    def save_results(
        self,
        accuracy: float,
        results: List[EvaluationResult],
        output_file: str
    ):
        """Sauvegarde les résultats d'évaluation"""
        output = {
            "accuracy": accuracy,
            "total_samples": len(results),
            "correct": sum(1 for r in results if r.correct),
            "results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "expected": r.expected_answer,
                    "predicted": r.predicted_answer,
                    "correct": r.correct,
                    "response": r.full_response
                }
                for r in results
            ]
        }
        
        filepath = LOGS_DIR / output_file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"Résultats sauvegardés: {filepath}")


def run_evaluation():
    """Compare les performances avant/après distillation"""
    print("=" * 60)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 60)
    
    num_samples = 50  
    
    print("\n--- Modèle de base (sans distillation) ---")
    base_evaluator = ModelEvaluator(adapter_path=None)
    base_accuracy, base_results = base_evaluator.evaluate_dataset(
        split="validation",
        num_samples=num_samples
    )
    base_evaluator.save_results(base_accuracy, base_results, "eval_base_model.json")
    print(f"Accuracy modèle de base: {base_accuracy * 100:.2f}%")
    
    del base_evaluator
    torch.cuda.empty_cache()
    
    stage1_path = CHECKPOINTS_DIR / "csqa_stage1"
    if stage1_path.exists():
        print("\n--- Modèle distillé (Stage 1) ---")
        stage1_evaluator = ModelEvaluator(adapter_path=str(stage1_path))
        stage1_accuracy, stage1_results = stage1_evaluator.evaluate_dataset(
            split="validation",
            num_samples=num_samples
        )
        stage1_evaluator.save_results(stage1_accuracy, stage1_results, "eval_stage1.json")
        print(f"Accuracy Stage 1: {stage1_accuracy * 100:.2f}%")
        
        del stage1_evaluator
        torch.cuda.empty_cache()
    else:
        print(f"\nAdapter Stage 1 non trouvé: {stage1_path}")
        stage1_accuracy = None
    
    stage2_path = CHECKPOINTS_DIR / "csqa_stage2"
    if stage2_path.exists():
        print("\n--- Modèle distillé (Stage 2) ---")
        stage2_evaluator = ModelEvaluator(adapter_path=str(stage2_path))
        stage2_accuracy, stage2_results = stage2_evaluator.evaluate_dataset(
            split="validation",
            num_samples=num_samples
        )
        stage2_evaluator.save_results(stage2_accuracy, stage2_results, "eval_stage2.json")
        print(f"Accuracy Stage 2: {stage2_accuracy * 100:.2f}%")
        
        del stage2_evaluator
    else:
        print(f"\nAdapter Stage 2 non trouvé: {stage2_path}")
        stage2_accuracy = None
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES RÉSULTATS")
    print("=" * 60)
    print(f"Modèle de base:    {base_accuracy * 100:.2f}%")
    if stage1_accuracy:
        print(f"Après Stage 1:     {stage1_accuracy * 100:.2f}% ({(stage1_accuracy - base_accuracy) * 100:+.2f}%)")
    if stage2_accuracy:
        print(f"Après Stage 2:     {stage2_accuracy * 100:.2f}% ({(stage2_accuracy - base_accuracy) * 100:+.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
