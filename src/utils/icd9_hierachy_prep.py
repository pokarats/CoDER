#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: WIP: intermediate file for ICD9 hierachy.
             The logger writes to a hidden folder './log/'
             and uses the name of this file, followed by the date (by default). The argument parser comes with a default
             option --quiet to keep the stdout clean. (if run from main with logger config, otherwise cout.txt in
             Sacred logging).


@author: Noon Pokaratsiri Goldstein
"""
# TODO: generate a .tsv file to store icd9 tree hierachy that will be used by icd9_tree.py
#   - read icd9_umls2020aa
#   - substitue descriptions with the ones from ICD9_descriptions
#   - validate hierachy by querying some randome ones to umls database
#   - write out hierachy in ICD9_Name TAB UMLS ICD9_Code TAB Tree Depth
