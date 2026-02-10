import torch
import sys

def check_gpu():
    print("=== Vérification de l'environnement GPU ===")
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("✅ CUDA est disponible!")
        device_count = torch.cuda.device_count()
        print(f"Nombre de GPUs détectés: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            
            # Vérifier la VRAM
            vram_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  VRAM Totale: {vram_total:.2f} GB")
            
            # Vérifier Compute Capability pour bitsandbytes (nécessite 7.0+)
            cc_major, cc_minor = torch.cuda.get_device_capability(i)
            print(f"  Compute Capability: {cc_major}.{cc_minor}")
            
            if cc_major < 7:
                print("  ⚠️ ATTENTION: Ce GPU peut être trop ancien pour bitsandbytes (4-bit quantization).")
            else:
                print("  ✅ Compatible avec bitsandbytes 4-bit.")
                
    else:
        print("❌ CUDA n'est PAS détecté. Le code tournera sur CPU (très lent).")
        print("Vérifiez l'installation de PyTorch avec CUDA : https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    check_gpu()
