import unittest
import dgl
import itertools
from src.utils.config import PROJ_FOLDER
from src.utils.corpus_readers import get_data
from src.utils.prepare_gnn_data import GNNDataReader, GNNDataset

DATA_DIR = f"{PROJ_FOLDER / 'data' / 'mimic3'}"
LINKED_DATA_DIR = f"{PROJ_FOLDER / 'data' / 'linked_data' / 'dummy'}"


class TestGNNData(unittest.TestCase):
    def setUp(self) -> None:
        self.data_reader, self.tr, self.dev, self.test = get_data(batch_size=2,
                                                                  dataset_class=GNNDataset,
                                                                  collate_fn=GNNDataset.collate_gnn,
                                                                  reader=GNNDataReader,
                                                                  data_dir=DATA_DIR,
                                                                  version="dummy",
                                                                  input_type="umls",
                                                                  prune_cui=True,
                                                                  cui_prune_file="full_cuis_to_discard_snomedcase4.pickle",
                                                                  vocab_fn="processed_full_umls_pruned.json")

        # get_dataloader returns a tuple of dataloader, embedding size, and num lasses for DGLDataset type
        self.tr_dl, self.tr_emb_size, self.tr_nclasses = self.tr
        self.dev_dl, self.dev_emb_size, self.dev_nclasses = self.dev
        self.test_dl, self.test_emb_size, self.test_nclasses = self.test

    def test_data_reader(self):
        for partition in ["train", "dev", "test"]:
            with self.subTest(partition=partition):
                self.assertIsNotNone(self.data_reader.get_dataset(partition))
                self.assertIsNotNone(self.data_reader.get_dataset_stats(partition))

    def test_emb_size(self):
        """
        All partitions should end up with the same embedding size, i.e. in our dataset == 100
        """
        for partition in [self.tr_emb_size, self.dev_emb_size, self.test_emb_size]:
            with self.subTest(partition=partition):
                self.assertEqual(partition, 100)

    def test_num_classes(self):
        """
        All partitions should have the same number of classes
        """

        def pairwise(iterable):
            # pairwise('ABCDEFG') --> AB BC CD DE EF FG
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)

        for nclasses_a, nclasses_b in pairwise([self.tr_nclasses, self.dev_nclasses, self.test_nclasses]):
            with self.subTest(nclasses_a=nclasses_a, nclasses_b=nclasses_b):
                self.assertEqual(nclasses_a, nclasses_b)

    def test_dataloader(self):
        for partition in [self.tr_dl, self.dev_dl, self.test_dl]:
            with self.subTest(partition=partition):
                g_batch, labels_batch = next(iter(partition))
                self.assertTrue(g_batch is not None and labels_batch.shape is not None)
                self.assertIsInstance(g_batch, dgl.DGLGraph)  # batch of graphs should still be dgl graphs
                self.assertLessEqual(labels_batch.shape[0], 2)  # num row should be less than or equal to batch size


if __name__ == '__main__':
    unittest.main(verbosity=2)
