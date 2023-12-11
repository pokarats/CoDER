import unittest
import dgl
import itertools
from src.utils.config import PROJ_FOLDER
from src.utils.corpus_readers import ProcessedIterExtended
from src.utils.prepare_gnn_data import GNNDataReader, GNNDataset

DATA_DIR = f"{PROJ_FOLDER / 'data' / 'mimic3'}"
LINKED_DATA_DIR = f"{PROJ_FOLDER / 'data' / 'linked_data'}"
SEM_FILE = f"{PROJ_FOLDER / 'data' / 'umls' / 'semantic_info.csv'}"


class TestCUISemanticInfo(unittest.TestCase):
    def setUp(self) -> None:
        self.data_reader_50 = GNNDataReader(data_dir=DATA_DIR,
                                            version="50",
                                            input_type="umls",
                                            prune_cui=True,
                                            cui_prune_file="50_cuis_to_discard_snomednoex.pickle",
                                            vocab_fn="processed_full_umls_pruned.json")

        self.data_reader_full = GNNDataReader(data_dir=DATA_DIR,
                                              version="full",
                                              input_type="umls",
                                              prune_cui=True,
                                              cui_prune_file="full_cuis_to_discard_snomednoex.pickle",
                                              vocab_fn="processed_full_umls_pruned.json")
        self.cui2tui = dict()
        self.cui2sg = dict()
        for row in ProcessedIterExtended(SEM_FILE, header=True, delimiter="\t"):
            cui, tui, sg = row[1], row[2], row[4]
            self.cui2tui[cui] = tui
            self.cui2sg[cui] = sg

    def test_cui_in_tui(self):
        for partition in ["train", "dev", "test"]:
            with self.subTest(partition=partition):
                for doc_id, input_ids, input_tokens, glabel, n_nodes in self.data_reader_50.get_dataset(partition):
                    for cui in input_tokens:
                        self.assertIsNotNone(self.cui2tui.get(cui), f"failed CUI: {cui} in partition: {partition}")

    def test_cui_in_sg(self):
        for partition in ["train", "dev", "test"]:
            with self.subTest(partition=partition):
                for doc_id, input_ids, input_tokens, glabel, n_nodes in self.data_reader_full.get_dataset(partition):
                    for cui in input_tokens:
                        self.assertIsNotNone(self.cui2sg.get(cui), f"failed CUI: {cui} in partition: {partition}")


if __name__ == '__main__':
    unittest.main(verbosity=1)
