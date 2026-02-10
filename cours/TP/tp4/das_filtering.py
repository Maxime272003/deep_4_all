"""
Phase 4 : Implémentation du Divergence-Aware Sampling (DAS)
Filtre les exemples selon la divergence Teacher-Student
"""

import json
import torch
import numpy as np
import nltk
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import (
    STUDENT_MODEL_ID,
    DAS_P_TEACHER_THRESHOLD,
    DAS_DIVERGENCE_THRESHOLD,
    DAS_MIN_TEACHER_SENTENCES_RATIO,
    DATA_DIR,
    LOGS_DIR
)


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


@dataclass
class SentenceScore:
    """Score DAS pour une phrase"""
    sentence: str
    p_teacher: float
    p_student: float
    divergence: float
    sentence_type: str


@dataclass
class ExampleDASScore:
    """Score DAS global pour un exemple"""
    instruction: str
    response: str
    answer_key: str
    temperature: float
    sentence_scores: List[SentenceScore]
    teacher_sentence_ratio: float
    total_divergence: float
    keep: bool


class DASFilter:
    """Filtrage des exemples via Divergence-Aware Sampling"""
    
    def __init__(self, load_model: bool = True):
        """
        Initialise le filtre DAS
        
        Args:
            load_model: Si True, charge le modèle étudiant (nécessite GPU)
        """
        self.tokenizer = None
        self.model = None
        
        if load_model:
            self._load_student_model()
    
    def _load_student_model(self):
        """Charge le modèle étudiant"""
        print(f"Chargement du modèle étudiant: {STUDENT_MODEL_ID}...")
        
        if torch.cuda.is_available():
            print("GPU détecté! Utilisation de la quantification 4-bit (rapide)")
            device_map = "auto"
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    STUDENT_MODEL_ID,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Erreur lors du chargement 4-bit (bitsandbytes): {e}")
                print("Tentative de chargement standard sur GPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    STUDENT_MODEL_ID,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
        else:
            print("⚠️ AUCUN GPU DÉTECTÉ! Le modèle va tourner sur CPU (très lent).")
            print("Pour activer le GPU, installez PyTorch avec le support CUDA.")
            print("Désactivation de la quantification 4-bit (incompatible CPU).")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                STUDENT_MODEL_ID,
                device_map="cpu",  
                trust_remote_code=True,
                torch_dtype=torch.float32 
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            STUDENT_MODEL_ID, 
            trust_remote_code=True
        )
        
        self.model.eval()
        print(f"Modèle chargé sur: {self.model.device}")
    
    def _compute_student_logprobs(
        self, 
        prompt: str, 
        response: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule les log-probabilités du modèle étudiant
        
        Args:
            prompt: Le prompt utilisateur
            response: La réponse du teacher
            
        Returns:
            Tuple (token_logprobs, token_ids) pour la partie réponse
        """
        messages = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        full_input_str = prompt_str + response
        
        inputs = self.tokenizer(full_input_str, return_tensors="pt").to(self.model.device)
        
        prompt_tokens_len = len(self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        shift_logits = logits[0, :-1, :]
        shift_labels = inputs["input_ids"][0, 1:]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits, shift_labels)
        token_logprobs = -token_losses.cpu().numpy()
        
        response_logprobs = token_logprobs[prompt_tokens_len - 1:]
        response_token_ids = shift_labels[prompt_tokens_len - 1:].cpu().numpy()
        
        return response_logprobs, response_token_ids
    
    def _classify_sentence(
        self, 
        p_teacher: float, 
        p_student: float
    ) -> str:
        """
        Classifie une phrase selon les critères DAS
        
        Args:
            p_teacher: Probabilité moyenne du teacher
            p_student: Probabilité moyenne du student
            
        Returns:
            Type de phrase: "teacher", "shared" ou "student"
        """
        divergence = p_teacher - p_student
        
        if p_teacher > DAS_P_TEACHER_THRESHOLD and divergence > DAS_DIVERGENCE_THRESHOLD:
            return "teacher"  
        elif abs(divergence) < DAS_DIVERGENCE_THRESHOLD:
            return "shared"   
        else:
            return "student"  
    
    def calculate_sentence_scores(
        self,
        prompt: str,
        response: str,
        teacher_logprobs: Optional[List[Dict]]
    ) -> List[SentenceScore]:
        """
        Calcule les scores DAS phrase par phrase
        
        Args:
            prompt: Le prompt utilisateur
            response: La réponse du teacher
            teacher_logprobs: Les logprobs du teacher (depuis l'API)
            
        Returns:
            Liste de scores par phrase
        """
        sentences = nltk.tokenize.sent_tokenize(response)
        
        if not sentences:
            return []
        
        if not teacher_logprobs:
            print("Warning: Pas de logprobs teacher, utilisation de valeurs estimées")
            teacher_probs = [0.8] * len(sentences)
        else:
            teacher_probs = self._align_teacher_probs(response, sentences, teacher_logprobs)
        
        student_logprobs, token_ids = self._compute_student_logprobs(prompt, response)
        student_probs = self._align_student_probs(response, sentences, student_logprobs, token_ids)
        
        scores = []
        for sent, p_t, p_s in zip(sentences, teacher_probs, student_probs):
            divergence = p_t - p_s
            sent_type = self._classify_sentence(p_t, p_s)
            
            scores.append(SentenceScore(
                sentence=sent,
                p_teacher=p_t,
                p_student=p_s,
                divergence=divergence,
                sentence_type=sent_type
            ))
        
        return scores
    
    def _align_teacher_probs(
        self,
        full_text: str,
        sentences: List[str],
        logprobs: List[Dict]
    ) -> List[float]:
        """Aligne les logprobs du teacher sur les phrases"""
        probs = []
        cursor = 0
        
        for sent in sentences:
            sent_logprobs = []
            current_accum = ""
            
            while cursor < len(logprobs):
                lp_data = logprobs[cursor]
                sent_logprobs.append(lp_data["logprob"])
                current_accum += lp_data["token"]
                cursor += 1
                
                if len(current_accum) >= len(sent):
                    break
            
            if sent_logprobs:
                p_mean = np.exp(np.mean(sent_logprobs))
            else:
                p_mean = 0.5  
            
            probs.append(float(p_mean))
        
        return probs
    
    def _align_student_probs(
        self,
        full_text: str,
        sentences: List[str],
        logprobs: np.ndarray,
        token_ids: np.ndarray
    ) -> List[float]:
        """Aligne les logprobs du student sur les phrases"""
        probs = []
        cursor = 0
        
        for sent in sentences:
            sent_logprobs = []
            current_accum = ""
            
            while cursor < len(token_ids):
                token_str = self.tokenizer.decode([token_ids[cursor]])
                sent_logprobs.append(logprobs[cursor])
                current_accum += token_str
                cursor += 1
                
                if len(current_accum) >= len(sent):
                    break
            
            if sent_logprobs:
                p_mean = np.exp(np.mean(sent_logprobs))
            else:
                p_mean = 0.5
            
            probs.append(float(p_mean))
        
        return probs
    
    def evaluate_example(
        self,
        example: Dict
    ) -> ExampleDASScore:
        """
        Évalue un exemple complet avec DAS
        
        Args:
            example: Dict avec instruction, response, answer_key, temperature, logprobs
            
        Returns:
            Score DAS global pour l'exemple
        """
        sentence_scores = self.calculate_sentence_scores(
            prompt=example["instruction"],
            response=example["response"],
            teacher_logprobs=example.get("logprobs")
        )
        
        if not sentence_scores:
            return ExampleDASScore(
                instruction=example["instruction"],
                response=example["response"],
                answer_key=example["answer_key"],
                temperature=example["temperature"],
                sentence_scores=[],
                teacher_sentence_ratio=0.0,
                total_divergence=0.0,
                keep=False
            )
        
        teacher_sentences = [s for s in sentence_scores if s.sentence_type == "teacher"]
        teacher_ratio = len(teacher_sentences) / len(sentence_scores)
        total_divergence = sum(s.divergence for s in sentence_scores)
        
        keep = teacher_ratio >= DAS_MIN_TEACHER_SENTENCES_RATIO
        
        return ExampleDASScore(
            instruction=example["instruction"],
            response=example["response"],
            answer_key=example["answer_key"],
            temperature=example["temperature"],
            sentence_scores=sentence_scores,
            teacher_sentence_ratio=teacher_ratio,
            total_divergence=total_divergence,
            keep=keep
        )
    
    def filter_dataset(
        self,
        input_file: str,
        output_file: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filtre un dataset complet avec DAS
        
        Args:
            input_file: Fichier JSON d'entrée (format raw avec logprobs)
            output_file: Fichier JSON de sortie (format ShareGPT filtré)
            
        Returns:
            Tuple (exemples gardés, exemples rejetés)
        """
        print(f"Chargement de {input_file}...")
        with open(DATA_DIR / input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        kept = []
        rejected = []
        
        print(f"Filtrage DAS de {len(data)} exemples...")
        for example in tqdm(data, desc="DAS Filtering"):
            try:
                das_score = self.evaluate_example(example)
                
                result = {
                    "instruction": example["instruction"],
                    "response": example["response"],
                    "answer_key": example["answer_key"],
                    "das_score": {
                        "teacher_ratio": das_score.teacher_sentence_ratio,
                        "total_divergence": das_score.total_divergence,
                        "keep": das_score.keep
                    }
                }
                
                if das_score.keep:
                    kept.append(result)
                else:
                    rejected.append(result)
                    
            except Exception as e:
                print(f"Erreur DAS: {e}")
                kept.append({
                    "instruction": example["instruction"],
                    "response": example["response"],
                    "answer_key": example["answer_key"],
                    "das_score": {"error": str(e)}
                })
        
        sharegpt_data = [
            {
                "conversations": [
                    {"from": "human", "value": ex["instruction"]},
                    {"from": "gpt", "value": ex["response"]}
                ]
            }
            for ex in kept
        ]
        
        with open(DATA_DIR / output_file, "w", encoding="utf-8") as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
        
        print(f"Résultat: {len(kept)} gardés, {len(rejected)} rejetés")
        print(f"Sauvegardé dans {DATA_DIR / output_file}")
        
        return kept, rejected
    
    def plot_das_distribution(
        self,
        kept: List[Dict],
        rejected: List[Dict],
        output_file: str = "das_distribution.png"
    ):
        """
        Trace l'histogramme de distribution des scores DAS
        """
        kept_ratios = [
            ex["das_score"].get("teacher_ratio", 0) 
            for ex in kept 
            if "das_score" in ex and "teacher_ratio" in ex["das_score"]
        ]
        rejected_ratios = [
            ex["das_score"].get("teacher_ratio", 0) 
            for ex in rejected 
            if "das_score" in ex and "teacher_ratio" in ex["das_score"]
        ]
        
        plt.figure(figsize=(10, 6))
        plt.hist(kept_ratios, bins=20, alpha=0.7, label=f"Gardés ({len(kept_ratios)})", color="green")
        plt.hist(rejected_ratios, bins=20, alpha=0.7, label=f"Rejetés ({len(rejected_ratios)})", color="red")
        plt.axvline(x=DAS_MIN_TEACHER_SENTENCES_RATIO, color="black", linestyle="--", label="Seuil")
        plt.xlabel("Teacher Sentence Ratio")
        plt.ylabel("Nombre d'exemples")
        plt.title("Distribution des scores DAS")
        plt.legend()
        plt.savefig(LOGS_DIR / output_file, dpi=150)
        plt.close()
        print(f"Graphique sauvegardé: {LOGS_DIR / output_file}")


def run_das_filtering():
    """Exécute le filtrage DAS sur les datasets générés"""
    print("=" * 50)
    print("FILTRAGE DAS")
    print("=" * 50)
    
    das_filter = DASFilter(load_model=True)
    
    print("\n--- Stage 1 ---")
    kept1, rejected1 = das_filter.filter_dataset(
        "csqa_stage1_raw.json",
        "csqa_stage1_filtered.json"
    )
    
    print("\n--- Stage 2 ---")
    kept2, rejected2 = das_filter.filter_dataset(
        "csqa_stage2_raw.json",
        "csqa_stage2_filtered.json"
    )
    
    das_filter.plot_das_distribution(kept1, rejected1, "das_stage1.png")
    das_filter.plot_das_distribution(kept2, rejected2, "das_stage2.png")
    
    print("\n" + "=" * 50)
    print("FILTRAGE TERMINÉ")
    print("=" * 50)


if __name__ == "__main__":
    run_das_filtering()
