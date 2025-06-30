import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import os
import shutil
import sys

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now import the module to be tested
# Assuming torch and transformers are installed in the environment,
# no need to mock them at the sys.modules level for basic import.
# We will mock specific classes/functions as needed.
from modelling.train_model import load_processed_data, prepare_datasets, train_model, JavaneseHateSpeechDataset, NUM_LABELS

# Define a dummy tokenizer function for testing purposes if the real one can't load
def dummy_tokenizer_func(texts, truncation=True, padding=True, max_length=128):
    # Simplified tokenizer mock: returns a structure similar to what real tokenizer does
    input_ids = [[101] + [i+1 for i, _ in enumerate(str(text).split()[:max_length-2])] + [102] for text in texts] # Added str(text) for safety
    attention_mask = [[1] * len(ids) for ids in input_ids]
    # Pad to max_length
    for i in range(len(input_ids)):
        padding_length = max_length - len(input_ids[i])
        input_ids[i].extend([0] * padding_length)
        attention_mask[i].extend([0] * padding_length)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


class TestTrainingModule(unittest.TestCase):

    def setUp(self):
        # Create dummy data for testing
        self.dummy_data = {
            'processed_text': [
                "teks siji kanggo tes", "teks loro uga kanggo tes", "telu papat limo enem",
                "pitu wolu songo sepuluh", "sewelas rolas telulas", "patbelas limolas nembelas"
            ],
            'label': [0, 1, 0, 1, 2, 3] # NUM_LABELS = 4
        }
        self.dummy_df = pd.DataFrame(self.dummy_data)
        self.dummy_csv_path = "dummy_test_data.csv"
        self.dummy_df.to_csv(self.dummy_csv_path, index=False)

        self.model_output_dir = "test_model_output"

        # Ensure the real AutoTokenizer and AutoModelForSequenceClassification are mocked
        # from the perspective of train_model module
        self.patcher_tokenizer_load = patch('modelling.train_model.AutoTokenizer.from_pretrained')
        self.mock_auto_tokenizer = self.patcher_tokenizer_load.start()

        # Configure the mock tokenizer instance
        self.mock_tokenizer_instance = MagicMock()
        self.mock_tokenizer_instance.tokenize = lambda x: x.split() # Mocking the .tokenize() method if called directly
        self.mock_tokenizer_instance.side_effect = dummy_tokenizer_func # Make the mock instance itself callable with dummy_tokenizer_func
        self.mock_tokenizer_instance.save_pretrained = MagicMock() # Mock save_pretrained
        self.mock_auto_tokenizer.return_value = self.mock_tokenizer_instance

        # Patch AutoModelForSequenceClassification
        self.patcher_model_load = patch('modelling.train_model.AutoModelForSequenceClassification.from_pretrained')
        self.mock_auto_model = self.patcher_model_load.start()
        self.mock_model_instance = MagicMock()
        self.mock_auto_model.return_value = self.mock_model_instance

        # Patch Trainer
        self.patcher_trainer = patch('modelling.train_model.Trainer')
        self.mock_trainer_class = self.patcher_trainer.start()
        self.mock_trainer_instance = MagicMock()
        self.mock_trainer_instance.train = MagicMock()
        self.mock_trainer_instance.save_model = MagicMock()
        self.mock_trainer_class.return_value = self.mock_trainer_instance

        # Patch torch.tensor specifically within the 'modelling.train_model' module's scope
        # This lambda will be called when modelling.train_model.torch.tensor is accessed
        self.patcher_mt_torch_tensor = patch('modelling.train_model.torch.tensor', side_effect=lambda x, dtype=None: x)
        self.mock_mt_torch_tensor = self.patcher_mt_torch_tensor.start()


    def tearDown(self):
        if os.path.exists(self.dummy_csv_path):
            os.remove(self.dummy_csv_path)
        if os.path.exists(self.model_output_dir):
            shutil.rmtree(self.model_output_dir)
        self.patcher_tokenizer_load.stop()
        self.patcher_model_load.stop()
        self.patcher_trainer.stop()
        self.patcher_mt_torch_tensor.stop() # Stop the new patcher


    def test_load_processed_data_success(self):
        df = load_processed_data(self.dummy_csv_path)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), len(self.dummy_data['processed_text']))
        self.assertTrue('processed_text' in df.columns)
        self.assertTrue('label' in df.columns)

    def test_load_processed_data_file_not_found(self):
        df = load_processed_data("non_existent_file.csv")
        self.assertIsNone(df)

    def test_load_processed_data_missing_columns(self):
        bad_data = {'text': ["satu"], 'sentiment': [0]}
        bad_df = pd.DataFrame(bad_data)
        bad_df.to_csv(self.dummy_csv_path, index=False)
        df = load_processed_data(self.dummy_csv_path)
        self.assertIsNone(df)

    def test_prepare_datasets_success(self):
        # Ensure the global tokenizer in train_model is our mock
        with patch('modelling.train_model.tokenizer', self.mock_tokenizer_instance):
            train_ds, val_ds = prepare_datasets(self.dummy_df)
            self.assertIsNotNone(train_ds)
            self.assertIsNotNone(val_ds)
            # Basic check on dataset size (split is 80/20)
            # Total 6 samples -> train 4 (0.8*6=4.8, floor or ceil based on implementation, usually 4 or 5)
            # val 2 (or 1)
            # With random_state=42 and 6 samples, train_test_split gives 4 train, 2 val
            self.assertEqual(len(train_ds), 4)
            self.assertEqual(len(val_ds), 2)

            # Check structure of one item
            item = train_ds[0]
            self.assertIn('input_ids', item)
            self.assertIn('attention_mask', item)
            self.assertIn('labels', item)


    def test_prepare_datasets_invalid_labels(self):
         # Test with labels out of NUM_LABELS range (0-3)
        invalid_label_data = {
            'processed_text': ["teks satu", "teks dua"],
            'label': [0, 5] # 5 is an invalid label if NUM_LABELS is 4
        }
        invalid_df = pd.DataFrame(invalid_label_data)
        with patch('modelling.train_model.tokenizer', self.mock_tokenizer_instance):
            train_ds, val_ds = prepare_datasets(invalid_df)
            self.assertIsNone(train_ds) # Should fail due to invalid labels
            self.assertIsNone(val_ds)


    def test_train_model_success_mocked(self):
        # Create dummy datasets
        # Mock encodings and labels for JavaneseHateSpeechDataset
        dummy_encodings = {'input_ids': [[1,2,3]]*2, 'attention_mask': [[1,1,1]]*2}
        dummy_labels = [0,1]
        train_ds = JavaneseHateSpeechDataset(dummy_encodings, dummy_labels)
        val_ds = JavaneseHateSpeechDataset(dummy_encodings, dummy_labels)

        with patch('modelling.train_model.tokenizer', self.mock_tokenizer_instance):
            # Set a temporary output directory for this test
            test_specific_model_dir = os.path.join(self.model_output_dir, "specific_test")

            # Call train_model
            result_path = train_model(
                train_ds, val_ds,
                model_output_dir=test_specific_model_dir,
                num_train_epochs=1, # Minimal epochs for test
                per_device_train_batch_size=1
            )

            self.assertIsNotNone(result_path)
            self.assertEqual(result_path, test_specific_model_dir)

            # Check if Trainer was called and its methods
            self.mock_trainer_class.assert_called_once()
            self.mock_trainer_instance.train.assert_called_once()
            self.mock_trainer_instance.save_model.assert_called_with(test_specific_model_dir)
            self.mock_tokenizer_instance.save_pretrained.assert_called_with(test_specific_model_dir)

            # Check if output directory was created by save_model (mocked, so we check if os.makedirs was called by train_model)
            # In the real train_model, os.makedirs is called before trainer.save_model
            # So, we can check if the directory exists (if save_model was not mocked to prevent dir creation)
            # For robust mocking, we'd mock os.makedirs. Here, we assume it's fine.
            # For this test, since trainer.save_model is mocked, the directory might not be created by it.
            # The os.makedirs in train_model should be called though.
            # Let's ensure the call to save_model implies the directory would be ready.
            # If we want to test directory creation, we should mock os.makedirs.
            # For now, the check that save_model was called with the correct path is sufficient.


    def test_train_model_dataset_none(self):
        result_path = train_model(None, None, model_output_dir=self.model_output_dir)
        self.assertIsNone(result_path)


    def test_javanese_hate_speech_dataset(self):
        encodings = {'input_ids': [[101, 100, 102], [101, 200, 102]], 'attention_mask': [[1,1,1], [1,1,1]]}
        labels = [0, 1]
        dataset = JavaneseHateSpeechDataset(encodings, labels)

        self.assertEqual(len(dataset), 2)

        item0 = dataset[0]
        self.assertEqual(item0['input_ids'], [101, 100, 102]) # Due to torch.tensor mock, it's list
        self.assertEqual(item0['labels'], 0)

        item1 = dataset[1]
        self.assertEqual(item1['input_ids'], [101, 200, 102])
        self.assertEqual(item1['labels'], 1)

if __name__ == '__main__':
    # Unmock for direct script execution if necessary, or handle imports carefully
    # For unittest TestLoader, the mocks at the top of the file should work.
    if 'torch' in sys.modules and isinstance(sys.modules['torch'], MagicMock):
        del sys.modules['torch'] # Attempt to reload real torch if available
    if 'transformers' in sys.modules and isinstance(sys.modules['transformers'], MagicMock):
        del sys.modules['transformers']

    unittest.main()

# Clean up sys.modules changes if the test script is imported elsewhere, though typically not an issue for test files.
# Resetting sys.modules is generally tricky and best avoided if possible.
# The mocks are primarily for the TestLoader.
# If running this script directly and needing real torch/transformers, they should be importable.
# However, for CI/CD or environments where they might not be, the mocks are helpful.
# The __main__ block here is more for illustrative purposes.
# Typically, you'd run `python -m unittest tests.test_training`
