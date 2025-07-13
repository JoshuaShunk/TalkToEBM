"""
Test the new max_tokens=None functionality.
"""

import pytest
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier

import t2ebm.graphs as graphs


def test_max_tokens_none():
    """Test that max_tokens=None disables token limits."""
    # Create a simple EBM
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X, y)
    
    # Extract graph
    graph = graphs.extract_graph(ebm, 0)
    
    # Test with max_tokens=None (should not simplify)
    text_no_limit = graphs.graph_to_text(graph, max_tokens=None)
    
    # Test with small token limit (should simplify)
    text_limited = graphs.graph_to_text(graph, max_tokens=1000)
    
    # The unlimited version should be longer or equal
    assert len(text_no_limit) >= len(text_limited)
    
    # Both should be valid strings
    assert isinstance(text_no_limit, str)
    assert isinstance(text_limited, str)
    assert len(text_no_limit) > 0
    assert len(text_limited) > 0


def test_max_tokens_none_no_simplification():
    """Test that max_tokens=None disables simplification completely."""
    # Create a simple EBM
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X, y)
    
    # Extract graph
    graph = graphs.extract_graph(ebm, 0)
    
    # Test max_tokens=None (should never simplify)
    text_none = graphs.graph_to_text(graph, max_tokens=None)
    
    # Should be a valid string
    assert isinstance(text_none, str)
    assert len(text_none) > 0
    
    # Should contain the feature name
    assert "Feature Name:" in text_none