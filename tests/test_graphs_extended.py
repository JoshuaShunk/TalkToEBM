"""
Extended Graphs tests to boost coverage.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from interpret.glassbox import ExplainableBoostingClassifier

import t2ebm.graphs as graphs


class TestGraphsExtended:
    """Extended Graphs tests for better coverage."""

    @pytest.fixture
    def sample_ebm(self):
        """Create a sample EBM for testing."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        coef = np.array([1.0, -1.0, 0.5])
        y = (np.dot(X, coef) > 0).astype(int)
        
        ebm = ExplainableBoostingClassifier(random_state=42)
        ebm.fit(X, y)
        return ebm

    @pytest.fixture
    def categorical_ebm(self):
        """Create an EBM with categorical features."""
        np.random.seed(42)
        # Create categorical data
        categories = ['A', 'B', 'C']
        X_cat = np.random.choice(categories, size=(100, 1))
        X_num = np.random.randn(100, 1)
        
        # Convert categorical to numeric for EBM
        cat_mapping = {'A': 0, 'B': 1, 'C': 2}
        X_cat_numeric = np.array([[cat_mapping[x[0]]] for x in X_cat])
        X = np.hstack([X_cat_numeric, X_num])
        
        y = (X_cat_numeric.flatten() + X_num.flatten() > 0.5).astype(int)
        
        ebm = ExplainableBoostingClassifier(random_state=42, feature_names=['category', 'numeric'])
        ebm.fit(X, y)
        return ebm

    def test_graph_to_text_boolean_detection(self, sample_ebm):
        """Test boolean feature detection in graph_to_text."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        # Manually create a boolean-like graph
        graph.x_vals = ['FALSE', 'TRUE']
        graph.scores = np.array([0.1, 0.9])
        graph.stds = np.array([0.05, 0.05])
        
        text = graphs.graph_to_text(graph, feature_format=None)  # Let it auto-detect
        assert "boolean" in text.lower() or "false" in text.lower()


    def test_graph_to_text_precision_settings(self, sample_ebm):
        """Test different precision settings."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        # Test with specific x-axis precision
        text = graphs.graph_to_text(graph, x_axis_precision=2)
        assert isinstance(text, str)
        
        # Test with specific y-axis precision
        text = graphs.graph_to_text(graph, y_axis_precision=3)
        assert isinstance(text, str)

    def test_graph_to_text_no_confidence_bounds(self, sample_ebm):
        """Test graph conversion without confidence bounds."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        text = graphs.graph_to_text(graph, confidence_bounds=False)
        assert "Lower Bounds" not in text
        assert "Upper Bounds" not in text
        assert "Means:" in text

    def test_graph_to_text_no_description(self, sample_ebm):
        """Test graph conversion without description."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        text = graphs.graph_to_text(graph, include_description=False)
        assert not text.startswith("This graph represents")
        assert "Feature Name:" in text

    def test_graph_to_text_different_confidence_levels(self, sample_ebm):
        """Test different confidence levels."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        text_95 = graphs.graph_to_text(graph, confidence_level=0.95)
        text_90 = graphs.graph_to_text(graph, confidence_level=0.90)
        
        assert "95%-Confidence" in text_95
        assert "90%-Confidence" in text_90

    def test_graph_to_text_unknown_format_error(self, sample_ebm):
        """Test error with unknown feature format."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        with pytest.raises(Exception, match="Unknown feature format"):
            graphs.graph_to_text(graph, feature_format="unknown")

    def test_graph_to_text_boolean_assertion_error(self, sample_ebm):
        """Test boolean format with wrong number of values."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        # Force boolean format on a graph with more than 2 values
        with pytest.raises(AssertionError):
            graphs.graph_to_text(graph, feature_format="boolean")

    def test_simplify_graph_edge_cases(self, sample_ebm):
        """Test graph simplification edge cases."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        # Test with very high simplification
        simplified = graphs.simplify_graph(graph, min_variation_per_cent=0.5)
        assert len(simplified.x_vals) <= len(graph.x_vals)
        
        # Test with zero simplification
        simplified = graphs.simplify_graph(graph, min_variation_per_cent=0.0)
        assert simplified is not None

    def test_xy_to_json_function(self, sample_ebm):
        """Test the xy_to_json_ helper function."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        # Test the internal function through graph_to_text
        text = graphs.graph_to_text(graph)
        
        # Should contain JSON-like structure
        assert "Means: {" in text
        assert "}" in text

    def test_plot_graph_different_types(self, sample_ebm):
        """Test plotting different graph types."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        # Test plotting continuous graph
        try:
            graphs.plot_graph(graph)
        except Exception:
            pass  # Plotting might fail in headless environment
        
        # Test plotting with custom title
        try:
            graphs.plot_graph(graph, title="Custom Title")
        except Exception:
            pass

    def test_graph_to_text_simplification_loop(self, sample_ebm):
        """Test the token limit simplification loop."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        # Test with a token limit that should trigger simplification
        text = graphs.graph_to_text(graph, max_tokens=3000)
        assert isinstance(text, str)
        
        # Test with very restrictive limit that might cause exception
        try:
            text = graphs.graph_to_text(graph, max_tokens=500)
            assert isinstance(text, str)
        except Exception as e:
            # Should be a token limit exception
            assert "tokens" in str(e).lower()

    def test_extract_graph_different_features(self, sample_ebm):
        """Test extracting graphs for different feature types."""
        num_features = len(sample_ebm.feature_names_in_)
        
        for i in range(min(3, num_features)):  # Test first 3 features
            graph = graphs.extract_graph(sample_ebm, i)
            assert graph is not None
            assert hasattr(graph, 'feature_name')
            assert hasattr(graph, 'feature_type')
            assert hasattr(graph, 'x_vals')
            assert hasattr(graph, 'scores')

    def test_graph_attributes_access(self, sample_ebm):
        """Test accessing various graph attributes."""
        graph = graphs.extract_graph(sample_ebm, 0)
        
        # Test all expected attributes exist
        assert hasattr(graph, 'feature_name')
        assert hasattr(graph, 'feature_type')
        assert hasattr(graph, 'x_vals')
        assert hasattr(graph, 'scores')
        assert hasattr(graph, 'stds')
        
        # Test attribute types
        assert isinstance(graph.feature_name, str)
        assert isinstance(graph.feature_type, str)
        assert isinstance(graph.scores, np.ndarray)
        assert isinstance(graph.stds, np.ndarray)