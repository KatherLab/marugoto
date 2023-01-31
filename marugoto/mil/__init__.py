"""Attention-based deep multiple instance learning.

An implementation of

    arXiv:1802.04712
    Ilse, Maximilian, Jakub Tomczak, and Max Welling.
    "Attention-based deep multiple instance learning."
    International conference on machine learning. PMLR, 2018.
"""

from ._mil import *
from . import data
from . import helpers
from . import model

__author__ = "Marko van Treeck"
__copyright__ = "Copyright 2022, Kather Lab"
__license__ = "MIT"
__version__ = "0.4.0"
__maintainer__ = "Marko van Treeck"
__email__ = "mvantreeck@ukaachen.de"


__changelog__ = {
    "0.2.0": "Allow creation of bag from multiple h5s, implement training and deployment helper",
    "0.3.0": "Add cross-validation helper",
    "0.4.0": "Add multi-input",
}
