"""
Integration tests for t2ebm package.
"""

import numpy as np
import pytest
from interpret.glassbox import ExplainableBoostingClassifier

import t2ebm
import t2ebm.prompts as prompts
from t2ebm.testing import DummyChatModel


class TestIntegration:
    """Integration tests."""

    @pytest.fixture
    def sample_ebm(self):
        """Create a sample EBM for testing."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        coef = np.array([1.0, -1.0])
        y = (np.dot(X, coef) > 0).astype(int)

        ebm = ExplainableBoostingClassifier(random_state=42)
        ebm.fit(X, y)
        return ebm

    def test_full_pipeline_with_dummy_llm(self, sample_ebm):
        """Test the full pipeline with dummy LLM."""
        llm = DummyChatModel()

        # Test high-level functions
        result = t2ebm.describe_graph(llm, sample_ebm, 0)
        assert isinstance(result, str)
        assert len(result) > 0

        result = t2ebm.describe_ebm(llm, sample_ebm)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prompt_system_messages(self):
        """Test prompt system message generation."""
        msg = prompts.graph_system_msg()
        assert isinstance(msg, str)
        assert "expert" in msg.lower()

        msg = prompts.graph_system_msg("a data analyst")
        assert "data analyst" in msg

    def test_summarize_ebm_prompt(self):
        """Test EBM summarization prompt."""
        messages = prompts.summarize_ebm(
            feature_importances="feature1: 0.8\nfeature2: 0.2",
            graph_descriptions="Feature 1 shows positive trend",
            dataset_description="Test dataset",
            num_sentences=5,
        )
        assert isinstance(messages, list)
        assert len(messages) >= 2
        assert any("feature1" in str(msg) for msg in messages)

    def test_chain_of_thought_prompts(self):
        """Test chain-of-thought prompting."""
        graph_text = "Feature Name: test\nFeature Type: continuous"
        messages = prompts.describe_graph_cot(graph_text, num_sentences=3)

        assert isinstance(messages, list)
        assert len(messages) >= 3
        assert any(msg["role"] == "system" for msg in messages)
        assert any(msg["role"] == "user" for msg in messages)
        assert any(msg["role"] == "assistant" for msg in messages)

    def test_module_imports(self):
        """Test that all expected modules and functions are importable."""
        # Test main package imports
        assert hasattr(t2ebm, "describe_graph")
        assert hasattr(t2ebm, "describe_ebm")
        assert hasattr(t2ebm, "feature_importances_to_text")

        # Test submodule imports
        assert hasattr(t2ebm, "graphs")
        assert hasattr(t2ebm, "llm")
        assert hasattr(t2ebm, "prompts")
        assert hasattr(t2ebm, "utils")

        # Test version
        assert hasattr(t2ebm, "__version__")
        assert isinstance(t2ebm.__version__, str)

    def test_error_handling(self, sample_ebm):
        """Test error handling in various scenarios."""
        llm = DummyChatModel()

        # Test invalid feature index
        with pytest.raises((IndexError, ValueError)):
            t2ebm.describe_graph(llm, sample_ebm, 999)

        # Test invalid model type
        with pytest.raises((AttributeError, TypeError)):
            t2ebm.describe_graph(llm, "not_a_model", 0)
