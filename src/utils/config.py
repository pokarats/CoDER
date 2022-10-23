#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: WIP: Configure project variables and dot_env variables for API keys etc.

@copyright: Noon Pokaratsiri Goldxtein or licensors, as applicable.

@author: Noon Pokaratsiri Goldstein
"""
import os
import platform
import sys
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

if platform.system() != 'Darwin':
    sys.path.append(os.getcwd())  # only needed for slurm


PROJ_FOLDER = Path(__file__).resolve().parent.parent.parent
SAVED_FOLDER = PROJ_FOLDER / f"scratch/.log/{date.today():%y_%m_%d}/{Path(__file__).stem}"
MODEL_FOLDER = PROJ_FOLDER / "res" / f"{date.today():%y_%m_%d}"
DOTENV_PATH = PROJ_FOLDER / f"src/configs/.env.dev"

try:
    load_dotenv(str(DOTENV_PATH))
    DEV_API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
    DEV_UMLS_API_KEY = os.environ.get("UMLS_API_KEY")
    assert DEV_API_KEY is not None
    assert DEV_UMLS_API_KEY is not None
except AssertionError:
    # this most likely happen when running on slurm
    DOTENV_PATH = "/home/pokaratsiri/projects/CoDER/src/configs/.env.dev"
    print(f"Use hard coded path to .env.dev file instead at {DOTENV_PATH}")
    load_dotenv(DOTENV_PATH)
    DEV_API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
    DEV_UMLS_API_KEY = os.environ.get("UMLS_API_KEY")
    assert DEV_API_KEY is not None
    assert DEV_UMLS_API_KEY is not None
