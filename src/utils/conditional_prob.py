import os
import itertools
import logging
import math
from collections import Counter
from pathlib import Path

from src.utils.config import PROJ_FOLDER
from src.utils.corpus_readers import MimicCuiDocIter, ProcessedIterExtended


DX = ['DISO', 'ANAT', 'PHYS', 'PHEN', 'LIVB']
PROC = ['PROC', 'DEVI', 'ACTI']
LABS = ['CHEM', 'T034', 'T059']
CONC = ['CONC']

logger = logging.getLogger(__name__)


class CUIEHRProbModel:
    def __init__(self, version, mode, embedding_type, raw_dir=f"{PROJ_FOLDER / 'data'}", save_dir=None,
                 cui_prune_file=None, threshold=0.7):
        self.version = str(version)
        self.mode = mode
        self.embedding_type = embedding_type
        self.raw_dir = raw_dir
        self.save_dir = save_dir
        self.cui_prune_file = f"{Path(self.raw_dir) / 'linked_data' / self.version / cui_prune_file}" \
            if cui_prune_file is not None else self._prune_file_path()
        self.threshold = threshold
        self.cui2tui = dict()
        self.tui2sg = dict()
        self.cui2sg = dict()
        self._load_mapping_dicts()
        # unigram count by EHR structural component
        self.dx_counter = Counter()
        self.proc_counter = Counter()
        self.labs_counter = Counter()
        self.conc_counter = Counter()
        # co-occurrence: (dx_i, proc_i), (proc_i,lab_i), (conc_i, dx_i), and (conc_i, proc)
        self.dx_proc = dict()
        self.proc_lab = dict()
        self.conc_dx = dict()
        self.conc_proc = dict()
        self._make_cui_count_dicts()

    def _semantic_file_paths(self):
        mimic_sem_info_file = os.path.join(self.raw_dir, "mimic3", "semantic_types_mimic.txt")
        umls_sem_info_file = os.path.join(self.raw_dir, "umls", "semantic_info.csv")

        return mimic_sem_info_file, umls_sem_info_file

    def _umls_doc_file_path(self):
        return os.path.join(self.raw_dir, "linked_data", self.version, f"train_{self.version}_umls.txt")

    def _prune_file_path(self):
        return os.path.join(self.raw_dir, "linked_data", self.version,
                            f"{self.version}_cuis_to_discard_{self.embedding_type}.pickle")

    def _load_mapping_dicts(self):
        logger.info(f"Loading tui and sg mapping dict...")
        mimic_sem_info_path, umls_sem_info_path = self._semantic_file_paths()
        logger.info(f"mimic sem info from: {mimic_sem_info_path}\n"
                    f"umls sem info from: {umls_sem_info_path}")
        with open(mimic_sem_info_path, mode="r") as sf:
            for line in sf:
                if line:
                    sg, sg_desc, tui, st_desc = line.split("|")
                    self.tui2sg[tui] = sg

        # extending tui semantic info to include everything in UMLS
        logger.info(f"Sem info from UMLS...")
        for row in ProcessedIterExtended(umls_sem_info_path, header=True, delimiter="\t"):
            cui, tui, tui_desc, sg = row[1], row[2], row[3], row[4]
            self.cui2tui[cui] = tui
            self.tui2sg[tui] = sg
            self.cui2sg[cui] = sg

    def _count_cui_by_ngram(self, sent, dxs_set, procs_set, labs_set, concs_set):
        for cui_1, cui_2 in itertools.pairwise(sent):
            if cui_1 in dxs_set and cui_2 in procs_set:
                # proc cui after dx cui
                try:
                    self.dx_proc[cui_1][cui_2] += 1
                except KeyError:
                    if cui_1 not in self.dx_proc:
                        self.dx_proc[cui_1] = dict()
                    self.dx_proc[cui_1][cui_2] = 1
            elif cui_1 in procs_set and cui_2 in labs_set:
                # lab cui after proc cui
                try:
                    self.proc_lab[cui_1][cui_2] += 1
                except KeyError:
                    if cui_1 not in self.proc_lab:
                        self.proc_lab[cui_1] = dict()
                    self.proc_lab[cui_1][cui_2] = 1
            elif cui_1 in concs_set and cui_2 in dxs_set:
                # dx cui after conc cui
                try:
                    self.conc_dx[cui_1][cui_2] += 1
                except KeyError:
                    if cui_1 not in self.conc_dx:
                        self.conc_dx[cui_1] = dict()
                    self.conc_dx[cui_1][cui_2] = 1
            elif cui_1 in concs_set and cui_2 in procs_set:
                # proc cui after conc cui
                try:
                    self.conc_proc[cui_1][cui_2] += 1
                except KeyError:
                    if cui_1 not in self.conc_proc:
                        self.conc_proc[cui_1] = dict()
                    self.conc_proc[cui_1][cui_2] = 1
            else:
                # do not count
                pass

    def _count_cui_by_sg(self, unique_dxs, unique_procs, unique_labs, unique_concs):
        for dx, proc in itertools.product(unique_dxs, unique_procs):
            try:
                self.dx_proc[dx][proc] += 1
            except KeyError:
                if dx not in self.dx_proc:
                    self.dx_proc[dx] = dict()
                self.dx_proc[dx][proc] = 1
        for proc, lab in itertools.product(unique_procs, unique_labs):
            try:
                self.proc_lab[proc][lab] += 1
            except KeyError:
                if proc not in self.proc_lab:
                    self.proc_lab[proc] = dict()
                self.proc_lab[proc][lab] = 1
        for conc, dx in itertools.product(unique_concs, unique_dxs):
            try:
                self.conc_dx[conc][dx] += 1
            except KeyError:
                if conc not in self.conc_dx:
                    self.conc_dx[conc] = dict()
                self.conc_dx[conc][dx] = 1
        for conc, proc in itertools.product(unique_concs, unique_procs):
            try:
                self.conc_proc[conc][proc] += 1
            except KeyError:
                if conc not in self.conc_proc:
                    self.conc_proc[conc] = dict()
                self.conc_proc[conc][proc] = 1

    def _make_cui_count_dicts(self):
        umls_doc_file = self._umls_doc_file_path()
        prune_file_path = self.cui_prune_file
        logger.info(f"umls_doc_file from: {umls_doc_file}")
        logger.info(f"prune file from: {prune_file_path}")
        logger.info(f"Compiling cui count dicts...")
        umls_doc_iter = MimicCuiDocIter(umls_doc_file, self.threshold, True, prune_file_path)
        for _, doc, _ in umls_doc_iter:
            dxs = []
            procs = []
            labs = []
            concs = []
            for sent in doc:
                for cui in sent:
                    sg = self.cui2sg.get(cui, '')
                    tui = self.cui2tui.get(cui, '')
                    if sg in DX and tui not in LABS:
                        self.dx_counter.update([cui])
                        if cui not in dxs:
                            dxs.append(cui)
                    elif tui in LABS or sg in LABS:
                        self.labs_counter.update([cui])
                        if cui not in labs:
                            labs.append(cui)
                    elif sg in CONC:
                        self.conc_counter.update([cui])
                        if cui not in concs:
                            concs.append(cui)
                    elif sg in PROC and tui not in LABS:
                        self.proc_counter.update([cui])
                        if cui not in procs:
                            procs.append(cui)
                    else:
                        logger.warning(f"unknown cui: {cui} sem info? {sg} -- {tui}")
            self._count_cui_by_sg(dxs, procs, labs, concs)
        logger.info(f"count dicts by sg completed!")

    def get_prob(self, prior_cui, cui):
        if "by_ngram" not in self.mode:
            cui_sg = self.cui2sg.get(cui, 'OTHER')
            joint_count = 0.0
            count_prior_cui = 0.0

            if prior_cui in self.dx_proc:
                # dx proc prob
                joint_count = self.dx_proc[prior_cui].get(cui, 0.0)
                count_prior_cui = self.dx_counter.get(prior_cui, 0.0)
            elif prior_cui in self.proc_lab:
                # proc lab prob
                joint_count = self.proc_lab[prior_cui].get(cui, 0.0)
                count_prior_cui = self.proc_counter.get(prior_cui, 0.0)
            elif prior_cui in self.conc_dx and cui_sg in DX:
                # conc dx prob
                joint_count = self.conc_dx[prior_cui].get(cui, 0.0)
                count_prior_cui = self.conc_counter.get(prior_cui, 0.0)
            elif prior_cui in self.conc_proc and cui_sg in PROC:
                # conc proc prob
                joint_count = self.conc_proc[prior_cui].get(cui, 0.0)
                count_prior_cui = self.conc_counter.get(prior_cui, 0.0)
            else:
                # none of these scenarios, then probability will be 0
                return float(0.0)

            try:
                return float(joint_count) / float(count_prior_cui)
            except ZeroDivisionError:
                return float(0.0)
        else:
            # ngram probability approach
            raise NotImplementedError(f"{self.mode} currently not supported!!!")


if __name__ == '__main__':
    cui_plm = CUIEHRProbModel(version=50, mode="combined_kg_ehr", embedding_type="snomedcase4", cui_prune_file=None)
        # test prob: proc_lab == 0.287
        # test prob: conc_dx == 0.733
        # test prob: conc_proc == 0.498
        # test prob: dx_proc == 0.425
    assert math.isclose(cui_plm.get_prob("C0150521", "C2698261"), 0.28, rel_tol=0.1)
    assert math.isclose(cui_plm.get_prob("C0442808", "C0221106"), 0.73, rel_tol=0.1)
    assert math.isclose(cui_plm.get_prob("C0442808", "C0011900"), 0.4, rel_tol=0.1)
    assert math.isclose(cui_plm.get_prob("C0020459", "C0011900"), 0.42, rel_tol=0.1)
    assert math.isclose(cui_plm.get_prob("C0020459", "C2698261"), 0.0, rel_tol=0.1)
    print(f"CUIEHRProbModel tested ok!!!")
