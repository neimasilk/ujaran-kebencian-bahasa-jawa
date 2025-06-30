#!/usr/bin/env python3
"""
Environment Validation Script for Ujaran Kebencian Bahasa Jawa Project

This script validates that all required libraries are properly installed
and can be imported successfully.
"""

import sys
import importlib
from typing import List, Tuple

def check_library(library_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a library can be imported and get its version.
    
    Args:
        library_name: Name of the library for display
        import_name: Actual import name (if different from library_name)
    
    Returns:
        Tuple of (success: bool, version_info: str)
    """
    if import_name is None:
        import_name = library_name
    
    try:
        module = importlib.import_module(import_name)
        
        # Try to get version information
        version = "Unknown"
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        
        return True, f"{library_name}: {version}"
    
    except ImportError as e:
        return False, f"{library_name}: FAILED - {str(e)}"
    except Exception as e:
        return False, f"{library_name}: ERROR - {str(e)}"

def main():
    """
    Main validation function that checks all required libraries.
    """
    print("🔍 Environment Validation for Ujaran Kebencian Bahasa Jawa Project")
    print("=" * 70)
    print(f"Python Version: {sys.version}")
    print("=" * 70)
    
    # Define libraries to check
    libraries_to_check = [
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        ("Transformers", "transformers"),
        ("PyTorch", "torch"),
        ("TensorFlow", "tensorflow"),
        ("Google API Client", "googleapiclient"),
        ("Google Auth", "google.auth"),
        ("Beautiful Soup", "bs4"),
        ("Requests", "requests"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("Plotly", "plotly"),
        ("Jupyter Lab", "jupyterlab"),
        ("PyYAML", "yaml"),
        ("Python-dotenv", "dotenv"),
        ("tqdm", "tqdm"),
        ("pytest", "pytest")
    ]
    
    success_count = 0
    total_count = len(libraries_to_check)
    failed_libraries = []
    
    print("\n📦 Checking Required Libraries:")
    print("-" * 50)
    
    for library_name, import_name in libraries_to_check:
        success, message = check_library(library_name, import_name)
        
        if success:
            print(f"✅ {message}")
            success_count += 1
        else:
            print(f"❌ {message}")
            failed_libraries.append(library_name)
    
    print("-" * 50)
    print(f"\n📊 Summary: {success_count}/{total_count} libraries successfully imported")
    
    if success_count == total_count:
        print("\n🎉 SUCCESS: All required libraries are properly installed!")
        print("\n✅ Your environment is ready for the Ujaran Kebencian Bahasa Jawa project.")
        print("\n🚀 Next steps:")
        print("   1. Start developing your hate speech detection model")
        print("   2. Begin data collection and preprocessing")
        print("   3. Explore the notebooks/ directory for examples")
        return 0
    else:
        print(f"\n⚠️  WARNING: {total_count - success_count} libraries failed to import")
        print("\n❌ Failed libraries:")
        for lib in failed_libraries:
            print(f"   - {lib}")
        print("\n🔧 To fix missing libraries, run:")
        print("   pip install -r requirements.txt")
        print("\n💡 If issues persist, check:")
        print("   1. Conda environment is activated: conda activate ujaran")
        print("   2. Python version compatibility (3.11 recommended)")
        print("   3. Internet connection for package downloads")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)