import unittest
from src.utils.corpus_readers import ProcessedIter, MimicDocIter, MimicCuiDocIter
from src.utils.config import PROJ_FOLDER
import itertools


TEST_CSV_FILE = f"{PROJ_FOLDER / 'data/unit_test/train_3.csv'}"
TEST_UMLS_FILE = f"{PROJ_FOLDER / 'data/unit_test/train_3_umls.txt'}"


class TestCorpusReaders(unittest.TestCase):

    def test_mimic_iter_size(self):
        rows = ProcessedIter(TEST_CSV_FILE)
        mimic_docs = MimicDocIter(TEST_CSV_FILE)
        self.assertEqual(len(list(rows)), len(list(mimic_docs)))

    def test_umls_doc_iter_size(self):
        docs = MimicDocIter(TEST_CSV_FILE)
        umls_docs = MimicCuiDocIter(TEST_UMLS_FILE)
        self.assertEqual(len(list(docs)), len(list(umls_docs)))

    def test_read_same_file(self):
        doc_ids = MimicDocIter(TEST_CSV_FILE, 0)
        doc_labs = MimicDocIter(TEST_CSV_FILE, 3)
        for d_id, d_lab in zip(doc_ids, doc_labs):
            with self.subTest(d_id=d_id, d_lab=d_lab):
                self.assertTrue(d_id is not None and d_lab is not None)

    def test_zip_iter(self):
        docs = MimicDocIter(TEST_CSV_FILE)
        umls_docs = MimicCuiDocIter(TEST_UMLS_FILE)
        d_ids, d_labs, d_lens, u_ids, u_docs, u_lens = [], [], [], [], [], []
        for (d_id, d_doc, d_lab, d_len), (u_id, u_doc, u_len) in zip(docs, umls_docs):
            d_ids.append(d_id)
            d_labs.append(d_lab)
            d_lens.append(d_len)
            u_ids.append(u_id)
            u_docs.append(u_doc)
            u_lens.append(u_len)

        def pairwise(iterable):
            # pairwise('ABCDEFG') --> AB BC CD DE EF FG
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)

        for it_a, it_b in pairwise([u_ids, d_ids, u_docs, d_labs, u_lens, d_lens]):
            with self.subTest(it_a=it_a, it_b=it_b):
                self.assertEqual(len(it_a), len(it_b))


if __name__ == '__main__':
    unittest.main()
