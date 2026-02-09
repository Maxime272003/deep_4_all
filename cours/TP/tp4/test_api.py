"""
Script de test simple pour vérifier l'API Infomaniak
"""

import requests
import json
from config import INFOMANIAK_API_URL, INFOMANIAK_API_KEY, TEACHER_MODEL

def test_api():
    """Test simple de l'API"""
    print("=" * 60)
    print("TEST DE L'API INFOMANIAK")
    print("=" * 60)
    
    # Vérifier la clé API
    if not INFOMANIAK_API_KEY:
        print("❌ ERREUR: Clé API non définie dans .env")
        print("   Éditez le fichier .env et ajoutez votre clé API")
        return False
    
    print(f"✓ URL: {INFOMANIAK_API_URL}")
    print(f"✓ Modèle: {TEACHER_MODEL}")
    print(f"✓ Clé API: {INFOMANIAK_API_KEY[:10]}...")
    
    # Préparer la requête
    headers = {
        "Authorization": f"Bearer {INFOMANIAK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": TEACHER_MODEL,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2? Answer in one sentence."
            }
        ]
    }
    
    print("\n📤 Envoi de la requête test...")
    
    try:
        response = requests.post(
            INFOMANIAK_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📥 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"\n✅ SUCCÈS!")
            print(f"Réponse: {content}")
            print("\n" + "=" * 60)
            print("L'API fonctionne correctement!")
            print("Vous pouvez lancer: python main.py generate")
            print("=" * 60)
            return True
        else:
            print(f"\n❌ ERREUR HTTP {response.status_code}")
            print(f"Réponse: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ ERREUR: Timeout (l'API ne répond pas)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERREUR: {e}")
        return False
    except Exception as e:
        print(f"❌ ERREUR inattendue: {e}")
        return False


if __name__ == "__main__":
    test_api()
