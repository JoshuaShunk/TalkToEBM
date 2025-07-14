"""
Test high-level functions.
"""

import numpy as np
import pytest
from interpret.glassbox import ExplainableBoostingClassifier

import t2ebm
import t2ebm.functions as functions
from t2ebm.testing import DummyChatModel


class TestHighLevelFunctions:
    """Test high-level API functions."""

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

    def test_describe_graph_with_dummy_llm(self, sample_ebm):
        """Test describe_graph with dummy LLM."""
        llm = DummyChatModel()

        result = functions.describe_graph(llm, sample_ebm, 0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_graph_with_parameters(self, sample_ebm):
        """Test describe_graph with additional parameters."""
        llm = DummyChatModel()

        result = functions.describe_graph(
            llm,
            sample_ebm,
            0,
            graph_description="Test graph description",
            dataset_description="Test dataset",
            task_description="Test task",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_ebm_with_dummy_llm(self, sample_ebm):
        """Test describe_ebm with dummy LLM."""
        llm = DummyChatModel()

        result = functions.describe_ebm(llm, sample_ebm)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_ebm_with_parameters(self, sample_ebm):
        """Test describe_ebm with additional parameters."""
        llm = DummyChatModel()

        result = functions.describe_ebm(
            llm,
            sample_ebm,
            dataset_description="Test dataset",
            y_axis_description="Test y-axis",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_feature_importances_to_text(self, sample_ebm):
        """Test feature_importances_to_text function."""
        result = functions.feature_importances_to_text(sample_ebm)
        assert isinstance(result, str)
        assert len(result) > 0
        # Check that feature names and importance values are present
        assert "feature_0000:" in result
        assert "feature_0001:" in result

    def test_invalid_feature_index(self, sample_ebm):
        """Test describe_graph with invalid feature index."""
        llm = DummyChatModel()

        with pytest.raises((IndexError, ValueError)):
            functions.describe_graph(llm, sample_ebm, 999)

    def test_describe_graph_string_llm(self, sample_ebm):
        """Test describe_graph with string LLM (should fail without API key)."""
        # This should attempt to use OpenAI and fail without proper setup
        with pytest.raises((t2ebm.llm.OpenAIInitializationError, Exception)):
            functions.describe_graph("gpt-3.5-turbo", sample_ebm, 0)
