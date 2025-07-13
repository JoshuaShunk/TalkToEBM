"""
TalkToEBM: A Natural Language Interface to Explainable Boosting Machines
"""

# Import modules that need to be accessible
from . import graphs, llm, prompts, utils

# high-level functions
from .functions import (
    describe_ebm,
    describe_graph,
    feature_importances_to_text,
)
from .version import __version__
