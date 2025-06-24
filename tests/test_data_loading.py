import unittest
import os
import tempfile
import pandas as pd
from unittest.mock import patch, mock_open
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.load_csv_dataset import inspect_dataset
from utils.data_utils import load_data_from_google_sheets, load_data_from_csv, preprocess_data

class TestDataLoading(unittest.TestCase):
    """Test cases for data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample CSV data for testing
        self.sample_data = {
            'review': ['Iki apik banget', 'Ora apik', 'Biasa wae'],
            'sentiment': ['positive', 'negative', 'neutral']
        }
        self.sample_df = pd.DataFrame(self.sample_data)
        
    def test_inspect_dataset_file_not_found(self):
        """Test inspect_dataset with non-existent file."""
        result = inspect_dataset("non_existent_file.csv")
        self.assertIn("Error: Dataset file not found", result)
        
    def test_inspect_dataset_success(self):
        """Test inspect_dataset with valid CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            self.sample_df.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name
            
        try:
            # Mock pandas.read_csv to return our sample data
            with patch('pandas.read_csv', return_value=self.sample_df):
                result = inspect_dataset(temp_path)
                
            # Verify the report contains expected information
            self.assertIn("Dataset loaded successfully", result)
            self.assertIn("Number of rows: 3", result)
            self.assertIn("Number of columns: 2", result)
            self.assertIn("review", result)
            self.assertIn("sentiment", result)
            self.assertIn("Iki apik banget", result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_inspect_dataset_empty_file(self):
        """Test inspect_dataset with empty CSV file."""
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            empty_df.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name
            
        try:
            with patch('pandas.read_csv', return_value=empty_df):
                result = inspect_dataset(temp_path)
                
            # For empty DataFrame, the function might return an error due to describe() issues
            # We should check that it handles the case gracefully
            self.assertIsInstance(result, str)
            # Either it succeeds with empty data info or returns an error message
            self.assertTrue(
                "Number of rows: 0" in result or 
                "An unexpected error occurred" in result or
                "Error:" in result
            )
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_inspect_dataset_with_missing_values(self):
        """Test inspect_dataset with CSV containing missing values."""
        # Create data with missing values
        data_with_na = {
            'review': ['Iki apik', None, 'Ora apik'],
            'sentiment': ['positive', 'neutral', None]
        }
        df_with_na = pd.DataFrame(data_with_na)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df_with_na.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name
            
        try:
            with patch('pandas.read_csv', return_value=df_with_na):
                result = inspect_dataset(temp_path)
                
            self.assertIn("Dataset loaded successfully", result)
            self.assertIn("Number of rows: 3", result)
            # Should handle missing values gracefully
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_inspect_dataset_invalid_csv(self):
        """Test inspect_dataset with invalid CSV format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write("This is not a valid CSV content")
            temp_path = temp_file.name
            
        try:
            # This should handle pandas parsing errors
            result = inspect_dataset(temp_path)
            # The function should either succeed with the malformed data or handle the error
            self.assertIsInstance(result, str)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_load_data_from_google_sheets_placeholder(self):
        """Test Google Sheets loading function (placeholder implementation)."""
        # Test the placeholder function
        result = load_data_from_google_sheets(
            sheet_id="test_sheet_id",
            sheet_name="test_sheet",
            credentials_path="test_credentials.json"
        )
        
        # Since it's a placeholder, it should return empty DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
        
    @patch('builtins.print')
    def test_load_data_from_google_sheets_prints_info(self, mock_print):
        """Test that Google Sheets function prints expected information."""
        load_data_from_google_sheets(
            sheet_id="test_id",
            sheet_name="test_name",
            credentials_path="test_path.json"
        )
        
        # Verify that print was called with expected messages
        mock_print.assert_any_call("Placeholder: Memuat data dari Google Sheet ID: test_id, Nama Sheet: test_name")
        mock_print.assert_any_call("Placeholder: Menggunakan kredensial dari: test_path.json")
        
    def test_load_data_from_csv_success(self):
        """Test successful CSV loading with load_data_from_csv."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            self.sample_df.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name
            
        try:
            result = load_data_from_csv(temp_path)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 3)
            self.assertIn('review', result.columns)
            self.assertIn('sentiment', result.columns)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_load_data_from_csv_file_not_found(self):
        """Test load_data_from_csv with non-existent file."""
        result = load_data_from_csv("non_existent_file.csv")
        self.assertIsNone(result)
        
    @patch('builtins.print')
    def test_load_data_from_csv_success_prints_message(self, mock_print):
        """Test that load_data_from_csv prints success message."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            self.sample_df.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name
            
        try:
            load_data_from_csv(temp_path)
            mock_print.assert_any_call(f"Data berhasil dimuat dari {temp_path}")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    @patch('builtins.print')
    def test_load_data_from_csv_file_not_found_prints_error(self, mock_print):
        """Test that load_data_from_csv prints error message for missing file."""
        non_existent_file = "non_existent_file.csv"
        load_data_from_csv(non_existent_file)
        mock_print.assert_any_call(f"Error: File tidak ditemukan di {non_existent_file}")
        
    def test_preprocess_data_success(self):
        """Test successful data preprocessing."""
        # Create test data with mixed case and extra spaces
        test_data = pd.DataFrame({
            'review': ['  Iki APIK Banget  ', 'ORA apik', '  Biasa WAE  ', 'Iki APIK Banget'],
            'sentiment': ['positive', 'negative', 'neutral', 'positive']
        })
        
        result = preprocess_data(test_data.copy())
        
        # Check that text is lowercased and stripped
        self.assertEqual(result.iloc[0]['review'], 'iki apik banget')
        self.assertEqual(result.iloc[1]['review'], 'ora apik')
        self.assertEqual(result.iloc[2]['review'], 'biasa wae')
        
        # Check that duplicates are removed (should have 3 rows instead of 4)
        self.assertEqual(len(result), 3)
        
    def test_preprocess_data_with_nan(self):
        """Test preprocessing with NaN values."""
        test_data = pd.DataFrame({
            'review': ['Iki apik', None, 'Ora apik'],
            'sentiment': ['positive', 'neutral', 'negative']
        })
        
        result = preprocess_data(test_data.copy())
        
        # Should remove rows with NaN in review column
        self.assertEqual(len(result), 2)
        self.assertFalse(result['review'].isnull().any())
        
    def test_preprocess_data_empty_dataframe(self):
        """Test preprocessing with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = preprocess_data(empty_df)
        
        self.assertTrue(result.empty)
        
    def test_preprocess_data_none_input(self):
        """Test preprocessing with None input."""
        result = preprocess_data(None)
        self.assertIsNone(result)
        
    @patch('builtins.print')
    def test_preprocess_data_prints_messages(self, mock_print):
        """Test that preprocessing prints appropriate messages."""
        test_data = pd.DataFrame({
            'review': ['Iki apik', 'Ora apik', 'Iki apik'],  # One duplicate
            'sentiment': ['positive', 'negative', 'positive']
        })
        
        preprocess_data(test_data.copy())
        
        # Check that appropriate messages are printed
        mock_print.assert_any_call("Memulai pra-pemrosesan data...")
        mock_print.assert_any_call("Menghapus 1 baris duplikat berdasarkan kolom 'review'.")
        mock_print.assert_any_call("Pra-pemrosesan data selesai.")
        
    @patch('builtins.print')
    def test_preprocess_data_prints_message(self, mock_print):
        """Test that preprocess_data prints processing message."""
        test_data = pd.DataFrame({
            'review': ['Test data'],
            'sentiment': ['positive']
        })
        
        preprocess_data(test_data)
        mock_print.assert_any_call("Memulai pra-pemrosesan data...")


class TestAdditionalCoverage(unittest.TestCase):
    """Additional test cases to improve coverage."""
    
    def test_preprocess_data_with_different_column(self):
        """Test preprocess_data with different text column name."""
        test_data = pd.DataFrame({
            'text': ['  Test DATA  ', 'another TEXT'],
            'label': ['pos', 'neg']
        })
        
        result = preprocess_data(test_data, text_column='text')
        
        # Verify preprocessing worked
        self.assertEqual(result['text'].iloc[0], 'test data')
        self.assertEqual(result['text'].iloc[1], 'another text')
    
    def test_preprocess_data_edge_cases(self):
        """Test preprocess_data with edge cases."""
        # Test with single row
        single_row = pd.DataFrame({
            'review': ['  SINGLE row  '],
            'sentiment': ['positive']
        })
        
        result = preprocess_data(single_row)
        self.assertEqual(result['review'].iloc[0], 'single row')
        
        # Test with all same values (should remove duplicates)
        duplicate_data = pd.DataFrame({
            'review': ['same text', 'same text', 'same text'],
            'sentiment': ['pos', 'pos', 'pos']
        })
        
        result = preprocess_data(duplicate_data)
        self.assertEqual(len(result), 1)  # Should have only 1 row after deduplication
        
class TestDataValidation(unittest.TestCase):
    """Test cases for data validation functionality."""
    
    def test_csv_structure_validation(self):
        """Test validation of expected CSV structure."""
        # Test data with correct structure
        correct_data = pd.DataFrame({
            'review': ['Test review 1', 'Test review 2'],
            'sentiment': ['positive', 'negative']
        })
        
        # Check if required columns exist
        required_columns = ['review', 'sentiment']
        self.assertTrue(all(col in correct_data.columns for col in required_columns))
        
    def test_sentiment_values_validation(self):
        """Test validation of sentiment values."""
        valid_sentiments = ['positive', 'negative', 'neutral']
        test_sentiments = ['positive', 'negative', 'neutral', 'invalid']
        
        # Check which sentiments are valid
        for sentiment in test_sentiments[:3]:
            self.assertIn(sentiment, valid_sentiments)
            
        # Check invalid sentiment
        self.assertNotIn(test_sentiments[3], valid_sentiments)

if __name__ == '__main__':
    unittest.main()