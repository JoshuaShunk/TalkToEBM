"""
TalkToEBM: A Natural Language Interface to Explainable Boosting Machines
"""

from .version import __version__

# Import modules that need to be accessible
from . import llm
from . import utils
from . import graphs
from . import prompts

# high-level functions
from .functions import (
    feature_importances_to_text,
    describe_graph,
    describe_ebm,
)
