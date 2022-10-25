#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: WIP: ICD9 hierachy Tree implementation.
             The logger writes to a hidden folder './log/'
             and uses the name of this file, followed by the date (by default). The argument parser comes with a default
             option --quiet to keep the stdout clean. (if run from main with logger config, otherwise cout.txt in
             Sacred logging).


@author: Noon Pokaratsiri Goldstein

This is an extension from SciSpacy UMLS Semantic Type Tree Implementation
"""
# TODO: create ICD9Tree Class from SemanticTypeTree Class with similar functionarlities
#   - function to look up node's parent
#   - function to look up node's children
#   - function to write hierachy in HEMKit format: ICD9_parent   ICD9_child
#   - need to map icd9 code and sklearn.Multilabel index, HEMKit format needs to also be written with this

from pathlib import Path
import sys
import os
from datetime import date
import time
import json
from typing import Optional, List, Any
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.umls_semantic_type_tree import UmlsSemanticTypeTree, SemanticTypeNode
from scispacy.file_cache import cached_path


class ICDSemanticTypeNode(SemanticTypeNode):
    icd9: str
    icd9_name: str
    icd9_children: List[Any]
    icd9_level: int


class ICD9SemanticTypeTree(UmlsSemanticTypeTree):
    def __int__(self, root: ICDSemanticTypeNode) -> None:
        super().__init__(root)
        children = self.get_children(root)
        children.append(root)
        # We'll store the nodes as a flattened list too, because
        # we don't just care about the leaves of the tree - sometimes
        # we'll need efficient access to intermediate nodes, and the tree
        # is tiny anyway.
        self.flat_icd9_nodes: List[ICDSemanticTypeNode] = children
        self.icd9_to_node = {node.icd9: node for node in self.flat_icd9_nodes}
        self.icd9_to_type_id = {node.icd9: node.type_id for node in self.flat_icd9_nodes}
        self.icd9_depth = max([node.level for node in self.flat_icd9_nodes])

    def get_node_from_id(self, icd9: str) -> SemanticTypeNode:
        return self.icd9_to_node[icd9]

    def get_type_id_from_icd9(self, icd9: str) -> str:
        return self.icd9_to_type_id[icd9]

    def get_canonical_name(self, icd9: str) -> str:
        return self.icd9_to_node[icd9].full_name

    def get_nodes_at_depth(self, level: int) -> List[SemanticTypeNode]:
        """
        Returns nodes at a particular depth in the tree.
        """
        return [node for node in self.flat_icd9_nodes if node.level == level]

    def get_children(self, node: SemanticTypeNode) -> List[ICDSemanticTypeNode]:
        """
        Recursively build up a flat list of all a node's children.
        """
        children = []
        for child in node.children:
            children.append(child)
            children.extend(self.get_children(child))
        return children

    def get_parent(self, node: SemanticTypeNode) -> Optional[SemanticTypeNode]:
        """
        Returns the parent of the input node, returning None if the input node is the root of the tree
        """
        current_depth = node.level
        possible_parents = self.get_nodes_at_depth(current_depth - 1)

        for possible_parent in possible_parents:
            for child in possible_parent.children:
                if child.type_id == node.type_id:
                    return possible_parent

        # If there are no parents, we are at the root and return None
        return None
