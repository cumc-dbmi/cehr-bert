import unittest
import tempfile
import shutil
import os
from pathlib import Path
from spark_apps.parameters import tokenizer_path, visit_tokenizer_path, bert_model_path
from trainers.train_bert_only import VanillaBertTrainer


class VanillaBertTrainerIntegrationTest(unittest.TestCase):
    def setUp(self):
        # Get the root folder of the project
        root_folder = Path(os.path.abspath(__file__)).parent.parent.parent.parent
        # Create a temporary directory to store model and tokenizer
        self.temp_dir = tempfile.mkdtemp()
        self.model_folder_path = os.path.join(self.temp_dir, 'model')
        Path(self.model_folder_path).mkdir(parents=True, exist_ok=True)

        self.tokenizer_path = os.path.join(self.model_folder_path, tokenizer_path)
        self.visit_tokenizer_path = os.path.join(self.model_folder_path, visit_tokenizer_path)
        self.model_path = os.path.join(self.model_folder_path, bert_model_path)
        self.tf_board_log_path = os.path.join(self.model_folder_path, 'logs')
        self.training_data_parquet_path = os.path.join(root_folder, 'sample_data/patient_sequence')

        self.embedding_size = 16
        self.context_window_size = 10
        self.depth = 6
        self.num_heads = 8
        self.include_visit_prediction = True
        self.include_prolonged_length_stay = False
        self.use_time_embedding = True
        self.use_behrt = False
        self.time_embeddings_size = 32
        self.batch_size = 16
        self.epochs = 1
        self.learning_rate = 0.001

        self.trainer = VanillaBertTrainer(
            training_data_parquet_path=self.training_data_parquet_path,
            tokenizer_path=self.tokenizer_path,
            visit_tokenizer_path=self.visit_tokenizer_path,
            embedding_size=self.embedding_size,
            context_window_size=self.context_window_size,
            depth=self.depth,
            num_heads=self.num_heads,
            include_visit_prediction=self.include_visit_prediction,
            include_prolonged_length_stay=self.include_prolonged_length_stay,
            use_time_embedding=self.use_time_embedding,
            use_behrt=self.use_behrt,
            time_embeddings_size=self.time_embeddings_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            model_path=self.model_path,
            tf_board_log_path=self.tf_board_log_path
        )

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_train_model(self):
        # TODO: Implement test for train_model method
        self.trainer.train_model()


if __name__ == '__main__':
    unittest.main()
