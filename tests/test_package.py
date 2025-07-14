"""
Automates unit testing.
"""

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier

import t2ebm


def test_package():
    from t2ebm.testing import DummyChatModel

    llm = DummyChatModel()
    # generate normally distributed data
    np.random.seed(0)
    X = np.random.randn(100, 5)
    # random coefficient and binary target
    coef = np.random.randn(5)
    y = (np.dot(X, coef) > 0).astype(int)
    ebm = ExplainableBoostingClassifier()
    ebm.fit(X, y)
    # high-level functions
    t2ebm.describe_graph(llm, ebm, 0)
    t2ebm.describe_ebm(llm, ebm)
    # graphs
    graph = t2ebm.graphs.extract_graph(ebm, 1)
    t2ebm.graphs.graph_to_text(graph)
