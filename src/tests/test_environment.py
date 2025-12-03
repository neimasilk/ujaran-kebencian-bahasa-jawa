"""Environment validation tests for Ujaran Kebencian Bahasa Jawa Project.

This module validates that all required libraries are properly installed
and can be imported successfully.

Author: AI Assistant
Date: 2025-01-01
"""

import unittest
import sys
import importlib
from typing import List, Tuple, Dict
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger

class TestEnvironment(unittest.TestCase):
    """Test cases for environment validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.logger = setup_logger("test_environment")
        
        # Define required libraries
        cls.required_libraries = [
            ("NumPy", "numpy"),
            ("Pandas", "pandas"),
            ("Scikit-learn", "sklearn"),
            ("Transformers", "transformers"),
            ("PyTorch", "torch"),
            ("Google API Client", "googleapiclient"),
            ("Google Auth", "google.auth"),
            ("Requests", "requests"),
            ("PyYAML", "yaml"),
            ("Python-dotenv", "dotenv"),
            ("tqdm", "tqdm")
        ]
        
        # Define optional libraries
        cls.optional_libraries = [
            ("TensorFlow", "tensorflow"),
            ("Beautiful Soup", "bs4"),
            ("Matplotlib", "matplotlib"),
            ("Seaborn", "seaborn"),
            ("Plotly", "plotly"),
            ("Jupyter Lab", "jupyterlab"),
            ("pytest", "pytest")
        ]
    
    def check_library(self, library_name: str, import_name: str = None) -> Tuple[bool, str, str]:
        """Check if a library can be imported and get its version.
        
        Args:
            library_name: Name of the library for display
            import_name: Actual import name (if different from library_name)
        
        Returns:
            Tuple of (success: bool, version_info: str, error_message: str)
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
            
            return True, version, ""
        
        except ImportError as e:
            return False, "Not installed", str(e)
        except Exception as e:
            return False, "Error", str(e)
    
    def test_python_version(self):
        """Test Python version compatibility."""
        self.logger.info(f"Testing Python version: {sys.version}")
        
        # Check Python version (should be 3.8+)
        major, minor = sys.version_info[:2]
        self.assertGreaterEqual(major, 3, "Python 3.x required")
        
        if major == 3:
            self.assertGreaterEqual(minor, 8, "Python 3.8+ recommended")
        
        self.logger.info(f"✅ Python {major}.{minor} is compatible")
    
    def test_required_libraries(self):
        """Test that all required libraries can be imported."""
        self.logger.info("Testing required libraries...")
        
        failed_libraries = []
        
        for library_name, import_name in self.required_libraries:
            with self.subTest(library=library_name):
                success, version, error = self.check_library(library_name, import_name)
                
                if success:
                    self.logger.info(f"✅ {library_name}: {version}")
                else:
                    self.logger.error(f"❌ {library_name}: {error}")
                    failed_libraries.append(library_name)
                
                self.assertTrue(success, f"{library_name} is required but not available: {error}")
        
        if not failed_libraries:
            self.logger.info("✅ All required libraries are available")
    
    def test_optional_libraries(self):
        """Test optional libraries (warnings only, not failures)."""
        self.logger.info("Testing optional libraries...")
        
        missing_optional = []
        
        for library_name, import_name in self.optional_libraries:
            success, version, error = self.check_library(library_name, import_name)
            
            if success:
                self.logger.info(f"✅ {library_name}: {version}")
            else:
                self.logger.warning(f"⚠️ {library_name}: Not available (optional)")
                missing_optional.append(library_name)
        
        if missing_optional:
            self.logger.info(f"Optional libraries not installed: {', '.join(missing_optional)}")
        else:
            self.logger.info("✅ All optional libraries are available")
    
    def test_google_drive_dependencies(self):
        """Test Google Drive specific dependencies."""
        self.logger.info("Testing Google Drive dependencies...")
        
        google_deps = [
            ("Google API Client", "googleapiclient"),
            ("Google Auth Transport", "google.auth.transport.requests"),
            ("Google OAuth", "google_auth_oauthlib.flow")
        ]
        
        all_available = True
        
        for library_name, import_name in google_deps:
            success, version, error = self.check_library(library_name, import_name)
            
            if success:
                self.logger.info(f"✅ {library_name}: Available")
            else:
                self.logger.error(f"❌ {library_name}: {error}")
                all_available = False
        
        if all_available:
            self.logger.info("✅ Google Drive integration dependencies are ready")
        else:
            self.logger.warning("⚠️ Some Google Drive dependencies are missing")
    
    def test_deepseek_dependencies(self):
        """Test DeepSeek API dependencies."""
        self.logger.info("Testing DeepSeek API dependencies...")
        
        deepseek_deps = [
            ("Requests", "requests"),
            ("JSON", "json"),
            ("Time", "time")
        ]
        
        for library_name, import_name in deepseek_deps:
            success, version, error = self.check_library(library_name, import_name)
            
            if success:
                self.logger.info(f"✅ {library_name}: Available")
            else:
                self.logger.error(f"❌ {library_name}: {error}")
                self.fail(f"DeepSeek dependency {library_name} not available: {error}")
        
        self.logger.info("✅ DeepSeek API dependencies are ready")
    
    def test_project_structure(self):
        """Test that project structure is correct."""
        self.logger.info("Testing project structure...")
        
        # Get project root (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        
        required_dirs = [
            "src",
            "src/config",
            "src/utils",
            "src/data_collection",
            "src/tests",
            "memory-bank",
            "vibe-guide"
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            self.assertTrue(full_path.exists(), f"Required directory not found: {dir_path}")
            self.logger.info(f"✅ Directory exists: {dir_path}")
        
        self.logger.info("✅ Project structure is correct")

def run_environment_check():
    """Run comprehensive environment check with detailed output."""
    # Setup logging
    logger = setup_logger("environment_check")
    
    logger.info("🔍 Environment Validation for Ujaran Kebencian Bahasa Jawa Project")
    logger.info("="*70)
    logger.info(f"Python Version: {sys.version}")
    logger.info("="*70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnvironment)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("ENVIRONMENT CHECK SUMMARY")
    logger.info("="*70)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        logger.info("\n🎉 SUCCESS: Environment is ready!")
        logger.info("\n✅ Your environment is ready for the Ujaran Kebencian Bahasa Jawa project.")
        logger.info("\n🚀 Next steps:")
        logger.info("   1. Start developing your hate speech detection model")
        logger.info("   2. Begin data collection and preprocessing")
        logger.info("   3. Run: python labeling.py to start labeling")
    else:
        logger.error("\n❌ Environment check failed!")
        logger.info("\n🔧 To fix issues:")
        logger.info("   1. Install missing packages: pip install -r requirements.txt")
        logger.info("   2. Check Python version (3.8+ recommended)")
        logger.info("   3. Verify virtual environment is activated")
        
        if result.failures:
            logger.error("\nFailures:")
            for test, traceback in result.failures:
                logger.error(f"  {test}: {traceback}")
        
        if result.errors:
            logger.error("\nErrors:")
            for test, traceback in result.errors:
                logger.error(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_environment_check()
    sys.exit(0 if success else 1)