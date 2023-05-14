import unittest
import dgl
from src.utils.config import PROJ_FOLDER
from src.utils.prepare_laat_data import get_data
from src.utils.prepare_gnn_data import GNNDataReader, GNNDataset


DATA_DIR = f"{PROJ_FOLDER / 'data' / 'mimic3'}"
LINKED_DATA_DIR = f"{PROJ_FOLDER / 'data' / 'linked_data' / 'dummy'}"


class TestGNNData(unittest.TestCase):
    def setUp(self) -> None:
        self.data_reader, self.tr_dl, self.dev_dl, self.test_dl = get_data(batch_size=2,
                                                                           dataset_class=GNNDataset,
                                                                           collate_fn=GNNDataset.collate_gnn,
                                                                           reader=GNNDataReader,
                                                                           data_dir=DATA_DIR,
                                                                           version="dummy",
                                                                           input_type="umls",
                                                                           prune_cui=True,
                                                                           cui_prune_file="full_cuis_to_discard_snomedcase4.pickle",
                                                                           vocab_fn="processed_full_umls_pruned.json")

    def test_data_reader(self):
        for partition in ["train", "dev", "test"]:
            with self.subTest(partition=partition):
                self.assertIsNotNone(self.data_reader.get_dataset(partition))
                self.assertIsNotNone(self.data_reader.get_dataset_stats(partition))

    def test_dataloader(self):
        for partition in [self.tr_dl, self.dev_dl, self.test_dl]:
            with self.subTest(partition=partition):
                g_batch, labels_batch = next(iter(partition))
                self.assertTrue(g_batch is not None and labels_batch.shape is not None)
                self.assertIsInstance(g_batch, dgl.DGLGraph)  # batch of graphs should still be dgl graphs
                self.assertLessEqual(labels_batch.shape[0], 2)  # num row should be less than or equal to batch size


if __name__ == '__main__':
    unittest.main(verbosity=2)
