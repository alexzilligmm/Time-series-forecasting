# src/enefit/config.py
import logging

import torch
import pandas as pd

pd.options.mode.chained_assignment = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)

