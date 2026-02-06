import pandas as pd
import random
import json
from pathlib import Path

def generate_dungeon_augmented(output_path: str, n_samples: int = 5000):
    """
    Génère un dataset augmenté pour Dungeon Oracle basé sur les règles secrètes.
    """
    
    # Vocabulaire (manuel pour éviter dépendance fichier)
    # Basé sur vocabulary_dungeon.json lu précédemment
    MOBS_WEAK = ["Rat", "Gobelin", "Orc", "Zombie"]
    MOBS_STRONG = ["Troll", "Hydre", "Minotaure"]
    BOSS = ["Dragon", "Liche", "Demon"]
    
    HEALS = ["Potion", "Grande_Potion", "Fontaine_Sacree", "Repas"]
    TRAPS = ["Piege_a_Pics", "Fosse", "Gaz_Toxique"]
    
    ITEMS_DEF = ["Amulette_Protection", "Armure_Ancienne", "Anneau_de_Vie"]
    ITEMS_OFF = ["Epee_Legendaire"]
    TREASURES = ["Coffre", "Or", "Gemmes"]
    
    FILLER = ["Couloir", "Salle_Vide", "Escalier", "Inscription"]
    
    data = []
    
    for _ in range(n_samples):
        sequence = ["Entree"]
        label = 1
        hp = 3  # Points de vie simulés
        has_amulet = False
        has_sword = False
        
        # Longueur aléatoire (parfois très longue pour généraliser)
        length = random.randint(10, 50)
        
        # Scenario construction
        events = []
        
        # Chance d'avoir l'amulette au début (Protection Boss)
        if random.random() < 0.2:
            events.append("Amulette_Protection")
            has_amulet = True
            
        # Chance d'avoir l'épée (Facilite combats)
        if random.random() < 0.2:
            events.append("Epee_Legendaire")
            has_sword = True
            
        # Remplissage
        for _ in range(length):
            roll = random.random()
            
            if roll < 0.3: # Filler
                events.append(random.choice(FILLER))
            elif roll < 0.5: # Trésor
                events.append(random.choice(TREASURES))
            elif roll < 0.6: # Soin
                heal = random.choice(HEALS)
                events.append(heal)
                hp = min(3, hp + 1) # Max 3 HP
            elif roll < 0.8: # Monstre faible
                mob = random.choice(MOBS_WEAK)
                events.append(mob)
                if not has_sword: # L'épée protège des petits mobs
                    hp -= 1
            elif roll < 0.9: # Piège
                trap = random.choice(TRAPS)
                events.append(trap)
                hp -= 1
            elif roll < 0.95: # Monstre fort
                mob = random.choice(MOBS_STRONG)
                events.append(mob)
                hp -= 2 # Très mal
            else: # Boss (Rare au milieu, mais possible)
                boss = random.choice(BOSS)
                events.append(boss)
                if not has_amulet:
                    hp = -10 # Mort instantanée sans amulette
        
        # Boss final (Classique)
        if random.random() < 0.5:
            events.append(random.choice(BOSS))
            if not has_amulet:
                 hp = -10
        
        # Logique Potion -> Dragon (Règle d'Or en priorité)
        # On injecte explicitement des patterns Potion/Dragon pour enseigner la règle
        if random.random() < 0.3:
            # Pattern Mortel : Dragon puis Potion
            events = ["Entree", "Dragon", "Potion", "Sortie"]
            label = 0
            # Force override
            hp = -999 
        elif random.random() < 0.3:
            # Pattern Survie : Potion puis Dragon (si pas déjà mort)
            events = ["Entree", "Potion", "Dragon", "Sortie"]
            label = 1
            hp = 1
            
        # Re-construire la séquence finale
        if hp > 0:
            label = 1
        else:
            label = 0
            
        # Si c'était un pattern explicite, on garde les events tels quels
        if hp != -999 and hp != 1:
            sequence.extend(events)
            sequence.append("Sortie")
        else:
            sequence = events
            
        data.append({
            "sequence": " -> ".join(sequence),
            "survived": label,
            "category": "synthetic"
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dataset augmenté généré : {output_path} ({len(df)} séquences)")

if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    output = data_dir / "train_dungeon_augmented.csv"
    generate_dungeon_augmented(output, n_samples=10000)
