import os

def clean_env_final():
    # Valid keys from previous successful run (Old Key 3, 4, 5)
    valid_keys = [
        "AIzaSyDGqjQxk_Loa64ZwtojlZoqLyLZSwhDVg4",
        "AIzaSyC-SZ7xKC-pExL1d1FuG2r6NLDySM-JwGo",
        "AIzaSyAbIZRQAU58a8loIu07wN4FsR_prM8cxFA"
    ]
    
    lines = []
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            lines = f.readlines()
            
    final_lines = []
    # Keep non-gemini lines
    for line in lines:
        if not line.strip().startswith('GEMINI_API_KEY'):
            final_lines.append(line)
            
    # Add re-indexed valid keys
    for i, key in enumerate(valid_keys):
        final_lines.append(f"GEMINI_API_KEY_{i+1}={key}\n")
        
    with open('.env', 'w') as f:
        f.writelines(final_lines)
    print("✅ .env clean: Key 1 & 2 removed, remaining keys re-indexed to 1-3.")

if __name__ == "__main__":
    clean_env_final()

