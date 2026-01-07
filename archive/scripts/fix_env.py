import os

def clean_env():
    # keys provided by user
    new_gemini_keys = {
        "GEMINI_API_KEY_1": "AIzaSyCEtKQ_-aY26nuIXgTPTgiS4S533gmpgIE",
        "GEMINI_API_KEY_2": "AIzaSyCcflRNqv6Vo3vxMQpJejd6uJXDtq2LUdg",
        "GEMINI_API_KEY_3": "AIzaSyDGqjQxk_Loa64ZwtojlZoqLyLZSwhDVg4",
        "GEMINI_API_KEY_4": "AIzaSyC-SZ7xKC-pExL1d1FuG2r6NLDySM-JwGo",
        "GEMINI_API_KEY_5": "AIzaSyAbIZRQAU58a8loIu07wN4FsR_prM8cxFA"
    }
    
    lines = []
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            lines = f.readlines()
            
    final_lines = []
    # Keep non-gemini lines
    for line in lines:
        if not line.strip().startswith('GEMINI_API_KEY'):
            final_lines.append(line)
            
    # Add new gemini keys
    for k, v in new_gemini_keys.items():
        final_lines.append(f"{k}={v}\n")
        
    with open('.env', 'w') as f:
        f.writelines(final_lines)
    print("✅ .env cleaned and updated with new keys.")

if __name__ == "__main__":
    clean_env()

