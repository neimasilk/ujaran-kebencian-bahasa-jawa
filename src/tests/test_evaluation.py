import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import os
import shutil
import json
import sys
import torch # Ditambahkan untuk mengatasi NameError

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from modelling.evaluate_model import (
    load_model_and_tokenizer,
    prepare_evaluation_data,
    predict,
    compute_metrics,
    save_evaluation_results,
    evaluate_model,
    EvaluationDataset
)

# Dummy tokenizer function similar to the one in test_training
def dummy_eval_tokenizer_func(texts, truncation=True, padding=True, max_length=128, return_tensors=None):
    input_ids = [[101] + [i+1 for i, _ in enumerate(str(text).split()[:max_length-2])] + [102] for text in texts]
    attention_mask = [[1] * len(ids) for ids in input_ids]
    for i in range(len(input_ids)): # Pad
        padding_length = max_length - len(input_ids[i])
        input_ids[i].extend([0] * padding_length)
        attention_mask[i].extend([0] * padding_length)

    encodings = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if return_tensors == "pt": # Handle return_tensors for PyTorch
        # In a real scenario, this would convert lists to torch.Tensor
        # For mocking, we ensure the structure is what the consuming code might expect (e.g., direct use in Dataset)
        # The EvaluationDataset class handles torch.tensor conversion if labels are not None
        # If labels is None (raw prediction), the EvaluationDataset is not used in the same way by `predict`
        # `predict` function's dataloader will handle batching of these lists if they are part of a dataset.
        # For direct use with model (if not using dataset/dataloader), tensors are needed.
        # Let's assume the mock structure is fine for EvaluationDataset to consume.
        pass # The mock dataset will handle tensor conversion if needed
    return encodings


class TestEvaluationModule(unittest.TestCase):

    def setUp(self):
        self.model_dir = "test_eval_model_dir"
        os.makedirs(self.model_dir, exist_ok=True)

        self.dummy_eval_data = {
            'text': ["teks evaluasi siji", "teks evaluasi loro", "teks evaluasi telu"],
            'label': [0, 1, 0]
        }
        self.dummy_eval_df = pd.DataFrame(self.dummy_eval_data)
        self.dummy_eval_csv_path = "dummy_test_eval_data.csv"
        self.dummy_eval_df.to_csv(self.dummy_eval_csv_path, index=False)

        # Mock AutoModelForSequenceClassification.from_pretrained
        self.patcher_model_load = patch('modelling.evaluate_model.AutoModelForSequenceClassification.from_pretrained')
        self.mock_auto_model_load = self.patcher_model_load.start()
        self.mock_model_instance = MagicMock()
        self.mock_model_instance.to = MagicMock(return_value=self.mock_model_instance) # For model.to(device)
        self.mock_model_instance.eval = MagicMock()
        # Mock model output (logits)
        # Simulate 3 texts, 4 labels. Logits shape (batch_size, num_labels)
        # Example: np.array([[0.1, 0.8, 0.05, 0.05], [0.7, 0.1, 0.1, 0.1], [0.2,0.1,0.6,0.1]]) -> preds [1,0,2]
        dummy_logits = np.array([[0.1, 0.8, 0.05, 0.05], [0.7, 0.1, 0.1, 0.1], [0.2,0.1,0.6,0.1]])
        # In torch, model output is an object with a 'logits' attribute
        mock_output = MagicMock()
        mock_output.logits = torch.tensor(dummy_logits) # Use real torch.tensor if available and not fully mocked
        self.mock_model_instance.return_value = mock_output # When model(...) is called
        self.mock_auto_model_load.return_value = self.mock_model_instance

        # Mock AutoTokenizer.from_pretrained
        self.patcher_tokenizer_load = patch('modelling.evaluate_model.AutoTokenizer.from_pretrained')
        self.mock_auto_tokenizer_load = self.patcher_tokenizer_load.start()
        self.mock_tokenizer_instance = MagicMock()
        self.mock_tokenizer_instance.side_effect = dummy_eval_tokenizer_func # Make it callable
        self.mock_auto_tokenizer_load.return_value = self.mock_tokenizer_instance

        # Mock torch.tensor in evaluate_model
        # This mock tensor should have a .to() method and return itself, and also be usable as a sequence/value
        # Mock torch.tensor in evaluate_model to return raw data for dataset items
        # This means EvaluationDataset.__getitem__ will return dict of lists/numbers
        self.patcher_ev_torch_tensor = patch('modelling.evaluate_model.torch.tensor', side_effect=lambda data, dtype=None: data)
        self.mock_ev_torch_tensor = self.patcher_ev_torch_tensor.start()


    def tearDown(self):
        if os.path.exists(self.dummy_eval_csv_path):
            os.remove(self.dummy_eval_csv_path)
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        self.patcher_model_load.stop()
        self.patcher_tokenizer_load.stop()
        self.patcher_ev_torch_tensor.stop()

    def test_load_model_and_tokenizer_success(self):
        model, tokenizer = load_model_and_tokenizer(self.model_dir)
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        self.mock_auto_model_load.assert_called_with(self.model_dir)
        self.mock_auto_tokenizer_load.assert_called_with(self.model_dir)

    def test_load_model_and_tokenizer_path_not_found(self):
        model, tokenizer = load_model_and_tokenizer("non_existent_dir")
        self.assertIsNone(model)
        self.assertIsNone(tokenizer)

    def test_prepare_evaluation_data_success(self):
        texts = ["contoh teks 1", "contoh teks 2"]
        labels = [0, 1]
        # Langsung gunakan self.mock_tokenizer_instance karena prepare_evaluation_data menerimanya sebagai argumen
        dataset = prepare_evaluation_data(texts, self.mock_tokenizer_instance, labels=labels)
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset), 2)
        item = dataset[0] # Calls __getitem__
        self.assertIn('input_ids', item)
        self.assertIn('labels', item)


    def test_predict_success(self):
        # Create a dummy EvaluationDataset
        # Encodings should be structured as expected by the dataset and model
        # Using dummy_eval_tokenizer_func to generate mock encodings
        mock_encodings = dummy_eval_tokenizer_func(self.dummy_eval_data['text'], return_tensors="pt") # return_tensors for predict

        # Ensure encodings are suitable for EvaluationDataset
        # The dummy_eval_tokenizer_func returns a dict of lists.
        # EvaluationDataset expects this structure.
        # torch.tensor is mocked to return the list itself. So dataset items are dicts of lists/numbers.

        dataset = EvaluationDataset(mock_encodings, self.dummy_eval_data['label'])

        # Mock DataLoader to yield batches directly, where each field is a mock tensor
        mock_batch_input_ids = MagicMock(name="batch_input_ids")
        mock_batch_input_ids.to.return_value = mock_batch_input_ids # .to(device)

        mock_batch_attn_mask = MagicMock(name="batch_attn_mask")
        mock_batch_attn_mask.to.return_value = mock_batch_attn_mask

        # Simulate one batch containing all data for simplicity in this test
        # The actual data for input_ids and attention_mask would be collated lists from dataset items
        # For the mock, we just need placeholders that can be passed to the model.
        # The actual content of mock_batch_input_ids.data doesn't strictly matter here as model is mocked.
        dummy_batch = {
            'input_ids': mock_batch_input_ids,
            'attention_mask': mock_batch_attn_mask
            # 'labels' are not used by `predict`'s model call, so not strictly needed in batch here
        }

        with patch('modelling.evaluate_model.torch.utils.data.DataLoader', return_value=[dummy_batch]) as mock_dataloader:
            predictions, raw_outputs = predict(self.mock_model_instance, dataset)

        mock_dataloader.assert_called_once_with(dataset, batch_size=8)
        self.mock_model_instance.assert_called_once_with(mock_batch_input_ids, attention_mask=mock_batch_attn_mask) # Perbaikan di sini
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(raw_outputs)
        self.assertEqual(len(predictions), 3)
        # Based on dummy_logits: [[0.1,0.8,0.05,0.05], [0.7,0.1,0.1,0.1], [0.2,0.1,0.6,0.1]]
        # Expected preds: [1, 0, 2]
        np.testing.assert_array_equal(predictions, np.array([1, 0, 2]))


    def test_compute_metrics_success(self):
        labels = np.array([0, 1, 0, 1, 2, 0])
        predictions = np.array([0, 1, 0, 0, 2, 1]) # 2 incorrect: pred[3] (0 vs 1), pred[5] (1 vs 0)

        metrics = compute_metrics(labels, predictions)
        self.assertIsNotNone(metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('confusion_matrix', metrics)

        expected_accuracy = (4/6) # 4 correct out of 6
        self.assertAlmostEqual(metrics['accuracy'], expected_accuracy)
        # For more precise checks on f1, precision, recall, you'd pre-calculate them.

    def test_compute_metrics_mismatch_len(self):
        labels = np.array([0,1,0])
        predictions = np.array([0,1])
        metrics = compute_metrics(labels, predictions)
        self.assertIsNone(metrics)


    @patch('builtins.open', new_callable=mock_open)
    @patch('modelling.evaluate_model.os.makedirs')
    def test_save_evaluation_results_success(self, mock_makedirs, mock_file_open):
        results = {'accuracy': 0.9, 'f1': 0.89}
        output_path = "test_results/eval.json"

        save_evaluation_results(results, output_path)

        mock_makedirs.assert_called_with(os.path.dirname(output_path), exist_ok=True)
        mock_file_open.assert_called_with(output_path, 'w', encoding='utf-8')
        # Check if json.dump was called (tricky with mock_open, check write)
        # Get the file handle from mock_open
        handle = mock_file_open()
        # Check if dump was called with the handle and results
        # This requires more advanced mocking of json.dump or checking handle.write calls
        # For simplicity, we check if open was called correctly.
        # To check content: json.dumps(results, ensure_ascii=False, indent=4)
        expected_json_string = json.dumps(results, ensure_ascii=False, indent=4)

        # Menggabungkan semua panggilan ke write dan membandingkan
        written_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)
        self.assertEqual(written_content, expected_json_string)


    @patch('modelling.evaluate_model.load_model_and_tokenizer')
    @patch('modelling.evaluate_model.prepare_evaluation_data')
    @patch('modelling.evaluate_model.predict')
    @patch('modelling.evaluate_model.compute_metrics')
    @patch('modelling.evaluate_model.save_evaluation_results')
    def test_evaluate_model_pipeline_with_file_and_labels(
        self, mock_save, mock_compute, mock_predict, mock_prepare, mock_load_model_tok
    ):
        # Setup mocks
        mock_load_model_tok.return_value = (MagicMock(), self.mock_tokenizer_instance) # Return a mock model and our tokenizer
        mock_dummy_dataset = MagicMock()
        mock_prepare.return_value = mock_dummy_dataset

        dummy_predictions = np.array([0, 1, 0])
        dummy_raw_outputs = np.array([[0.9,0.1],[0.1,0.9],[0.8,0.2]])
        mock_predict.return_value = (dummy_predictions, dummy_raw_outputs)

        dummy_metrics = {'accuracy': 1.0}
        mock_compute.return_value = dummy_metrics

        # Call the main function
        results = evaluate_model(
            model_path=self.model_dir,
            eval_data_path_or_texts=self.dummy_eval_csv_path,
            text_column='text',
            label_column='label',
            is_file=True
        )

        self.assertIsNotNone(results)
        mock_load_model_tok.assert_called_with(self.model_dir)

        # Check that pd.read_csv was implicitly called (prepare_evaluation_data should handle it)
        # We can check the call to prepare_evaluation_data
        # prepare_evaluation_data is called with list of texts and labels
        expected_texts = self.dummy_eval_df['text'].astype(str).tolist()
        expected_labels = self.dummy_eval_df['label'].tolist()
        mock_prepare.assert_called_with(expected_texts, self.mock_tokenizer_instance, labels=expected_labels)

        mock_predict.assert_called_with(mock_load_model_tok.return_value[0], mock_dummy_dataset)
        mock_compute.assert_called_with(expected_labels, dummy_predictions) # Ensure labels are passed correctly

        expected_final_results = {"predictions": dummy_predictions.tolist(), "raw_outputs_logits": dummy_raw_outputs.tolist()}
        expected_final_results.update(dummy_metrics)
        self.assertEqual(results, expected_final_results)

        final_output_path = os.path.join(self.model_dir, "evaluation_results.json")
        mock_save.assert_called_with(expected_final_results, final_output_path)


    @patch('modelling.evaluate_model.load_model_and_tokenizer')
    @patch('modelling.evaluate_model.prepare_evaluation_data')
    @patch('modelling.evaluate_model.predict')
    @patch('modelling.evaluate_model.save_evaluation_results')
    def test_evaluate_model_pipeline_raw_texts_no_labels(
        self, mock_save, mock_predict, mock_prepare, mock_load_model_tok
    ):
        raw_texts = ["teks mentah satu", "teks mentah dua"]
        mock_load_model_tok.return_value = (MagicMock(), self.mock_tokenizer_instance)
        mock_dummy_dataset = MagicMock()
        mock_prepare.return_value = mock_dummy_dataset
        dummy_predictions = np.array([0,1])
        dummy_raw_outputs = np.array([[0.7,0.3],[0.2,0.8]])
        mock_predict.return_value = (dummy_predictions, dummy_raw_outputs)

        results = evaluate_model(
            model_path=self.model_dir,
            eval_data_path_or_texts=raw_texts,
            is_file=False,
            output_file="raw_preds.json"
        )
        self.assertIsNotNone(results)
        mock_prepare.assert_called_with(raw_texts, self.mock_tokenizer_instance, labels=None)
        mock_predict.assert_called_with(mock_load_model_tok.return_value[0], mock_dummy_dataset)

        expected_results = {"predictions": dummy_predictions.tolist(), "raw_outputs_logits": dummy_raw_outputs.tolist()}
        self.assertEqual(results, expected_results)
        final_output_path = os.path.join(self.model_dir, "raw_preds.json")
        mock_save.assert_called_with(expected_results, final_output_path)


if __name__ == '__main__':
    unittest.main()
