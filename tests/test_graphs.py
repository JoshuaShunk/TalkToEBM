"""
Test graph extraction and conversion functionality.
"""

import numpy as np
import pytest
from interpret.glassbox import ExplainableBoostingClassifier

import t2ebm.graphs as graphs


class TestGraphs:
    """Test graph functionality."""

    @pytest.fixture
    def sample_ebm(self):
        """Create a sample EBM for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        coef = np.array([1.5, -0.8, 0.3])
        y = (np.dot(X, coef) > 0).astype(int)

        ebm = ExplainableBoostingClassifier(random_state=42)
        ebm.fit(X, y)
        return ebm

    def test_extract_graph_continuous(self, sample_ebm):
        """Test extracting a graph for a continuous feature."""
        graph = graphs.extract_graph(sample_ebm, 0)

        assert graph is not None
        assert hasattr(graph, "feature_type")
        assert hasattr(graph, "feature_name")
        assert hasattr(graph, "x_vals")
        assert hasattr(graph, "scores")

    def test_graph_to_text_basic(self, sample_ebm):
        """Test basic graph to text conversion."""
        graph = graphs.extract_graph(sample_ebm, 0)
        text = graphs.graph_to_text(graph)

        assert isinstance(text, str)
        assert len(text) > 0
        assert "Feature Name:" in text
        assert "Feature Type:" in text

    def test_graph_to_text_with_token_limit(self, sample_ebm):
        """Test graph to text conversion with token limit."""
        graph = graphs.extract_graph(sample_ebm, 0)
        # Use a reasonable token limit that won't cause exceptions
        text_limited = graphs.graph_to_text(graph, max_tokens=2000)
        text_unlimited = graphs.graph_to_text(graph, max_tokens=10000)

        assert len(text_limited) <= len(text_unlimited)

    def test_simplify_graph(self, sample_ebm):
        """Test graph simplification."""
        graph = graphs.extract_graph(sample_ebm, 0)
        simplified = graphs.simplify_graph(graph, min_variation_per_cent=0.1)

        assert simplified is not None
        # Simplified graph should generally have fewer points
        assert len(simplified.x_vals) <= len(graph.x_vals)

    def test_plot_graph_no_error(self, sample_ebm):
        """Test that plotting doesn't raise errors."""
        graph = graphs.extract_graph(sample_ebm, 0)

        # Should not raise an exception
        try:
            graphs.plot_graph(graph)
        except Exception as e:
            pytest.fail(f"plot_graph raised an exception: {e}")

    def test_extract_all_features(self, sample_ebm):
        """Test extracting graphs for all features."""
        num_features = len(sample_ebm.feature_names_in_)

        for i in range(num_features):
            graph = graphs.extract_graph(sample_ebm, i)
            assert graph is not None
            text = graphs.graph_to_text(graph)
            assert len(text) > 0
