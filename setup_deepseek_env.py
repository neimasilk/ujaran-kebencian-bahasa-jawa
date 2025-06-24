#!/usr/bin/env python3
"""
Script untuk setup environment variables yang aman untuk DeepSeek API.
Menghindari hardcoding API key dalam kode.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """
    Membuat file .env untuk menyimpan API key DeepSeek dengan aman.
    """
    env_file = Path(".env")
    
    if env_file.exists():
        print("File .env sudah ada.")
        response = input("Apakah Anda ingin mengupdate API key? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("=== SETUP DEEPSEEK API KEY ===")
    print("Masukkan API key DeepSeek Anda.")
    print("API key akan disimpan secara aman di file .env")
    print()
    
    api_key = input("DeepSeek API Key: ").strip()
    
    if not api_key:
        print("Error: API key tidak boleh kosong!")
        return False
    
    # Validasi format API key (basic check)
    if len(api_key) < 20:
        print("Warning: API key tampaknya terlalu pendek. Pastikan Anda memasukkan key yang benar.")
        confirm = input("Lanjutkan? (y/n): ")
        if confirm.lower() != 'y':
            return False
    
    # Buat atau update file .env
    env_content = f"""# DeepSeek API Configuration
# File ini berisi API key yang sensitif - JANGAN commit ke repository!

DEEPSEEK_API_KEY={api_key}

# Optional: DeepSeek API configuration
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.1
DEEPSEEK_MAX_TOKENS=200

# Rate limiting
DEEPSEEK_REQUESTS_PER_MINUTE=60
DEEPSEEK_DELAY_BETWEEN_REQUESTS=0.5
"""
    
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print(f"✅ API key berhasil disimpan di {env_file}")
        print()
        print("PENTING:")
        print("1. File .env sudah ditambahkan ke .gitignore")
        print("2. JANGAN share atau commit file .env ke repository")
        print("3. Backup API key Anda di tempat yang aman")
        
        return True
        
    except Exception as e:
        print(f"Error menyimpan file .env: {e}")
        return False

def update_gitignore():
    """
    Memastikan .env ada di .gitignore untuk keamanan.
    """
    gitignore_file = Path(".gitignore")
    
    # Baca existing .gitignore
    existing_content = ""
    if gitignore_file.exists():
        with open(gitignore_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()
    
    # Check apakah .env sudah ada di .gitignore
    if ".env" not in existing_content:
        # Tambahkan .env ke .gitignore
        additional_content = "\n# Environment variables (API keys, secrets)\n.env\n.env.local\n.env.*.local\n"
        
        with open(gitignore_file, 'a', encoding='utf-8') as f:
            f.write(additional_content)
        
        print("✅ .env ditambahkan ke .gitignore")
    else:
        print("✅ .env sudah ada di .gitignore")

def test_api_connection():
    """
    Test koneksi ke DeepSeek API dengan API key yang disimpan.
    """
    try:
        from dotenv import load_dotenv
        import requests
        
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            print("❌ API key tidak ditemukan di environment variables")
            return False
        
        # Test API call
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            "max_tokens": 10
        }
        
        print("Testing API connection...")
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ API connection successful!")
            print("DeepSeek API siap digunakan untuk pelabelan data.")
            return True
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return False
            
    except ImportError:
        print("❌ Module 'python-dotenv' atau 'requests' tidak ditemukan")
        print("Install dengan: pip install python-dotenv requests")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def show_usage_instructions():
    """
    Menampilkan instruksi penggunaan setelah setup.
    """
    print("\n=== CARA PENGGUNAAN ===")
    print("\n1. Untuk menjalankan pelabelan DeepSeek:")
    print("   python src/data_collection/deepseek_labeling.py")
    print()
    print("2. Atau secara programmatic:")
    print("   from dotenv import load_dotenv")
    print("   load_dotenv()")
    print("   api_key = os.getenv('DEEPSEEK_API_KEY')")
    print("   # Gunakan api_key untuk pelabelan")
    print()
    print("3. Untuk monitoring usage:")
    print("   - Check file log yang dihasilkan")
    print("   - Monitor biaya di DeepSeek dashboard")
    print()
    print("4. File output:")
    print("   - data/processed/deepseek_labeled_dataset.csv")
    print("   - data/processed/deepseek_validation_subset.csv")

def main():
    """
    Main function untuk setup DeepSeek environment.
    """
    print("🚀 SETUP DEEPSEEK V3 API UNTUK PELABELAN DATA")
    print("=" * 50)
    print()
    
    # Step 1: Create .env file
    if not create_env_file():
        print("Setup dibatalkan.")
        return
    
    print()
    
    # Step 2: Update .gitignore
    update_gitignore()
    
    print()
    
    # Step 3: Test API connection
    test_connection = input("Test koneksi API sekarang? (y/n): ")
    if test_connection.lower() == 'y':
        print()
        if test_api_connection():
            print("\n🎉 Setup berhasil! DeepSeek API siap digunakan.")
        else:
            print("\n⚠️  Setup selesai, tapi ada masalah dengan API connection.")
            print("Periksa API key dan koneksi internet Anda.")
    
    # Step 4: Show usage instructions
    show_usage_instructions()
    
    print("\n=== SETUP SELESAI ===")
    print("Anda sekarang dapat menggunakan DeepSeek API untuk pelabelan data.")

if __name__ == "__main__":
    main()