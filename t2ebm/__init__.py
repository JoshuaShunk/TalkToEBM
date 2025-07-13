"""
TalkToEBM: A Natural Language Interface to Explainable Boosting Machines
"""

from . import graphs, llm, prompts, utils
from .functions import describe_ebm, describe_graph, feature_importances_to_text
from .version import __version__
