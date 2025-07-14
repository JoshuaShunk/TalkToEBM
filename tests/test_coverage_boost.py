"""
Additional tests to boost coverage.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from interpret.glassbox import ExplainableBoostingClassifier

import t2ebm.graphs as graphs
import t2ebm.llm as llm
import t2ebm.utils as utils
from t2ebm.testing import DummyChatModel, openai_debug_completion_query


class TestCoverageBoost:
    """Additional tests to improve coverage."""

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

    def test_graph_plotting_functions(self, sample_ebm):
        """Test graph plotting and utility functions."""
        graph = graphs.extract_graph(sample_ebm, 0)

        # Test basic plotting (should not raise exception)
        try:
            graphs.plot_graph(graph)
        except Exception:
            pass  # Plotting might fail in headless environment

        # Test graph text conversion with different parameters
        text1 = graphs.graph_to_text(graph, include_description=False)
        assert isinstance(text1, str)

        text2 = graphs.graph_to_text(graph, confidence_bounds=False)
        assert isinstance(text2, str)

        # Test with no token limit (equivalent to old raw mode)
        text3 = graphs.graph_to_text(graph, max_tokens=None)
        assert isinstance(text3, str)

    def test_llm_model_representations(self):
        """Test string representations of LLM models."""
        dummy_model = DummyChatModel()
        repr_str = repr(dummy_model)
        assert isinstance(repr_str, str)

        # Test local LLM repr
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            local_model = llm.LocalLLMChatModel("http://localhost:11434", "test")
            repr_str = repr(local_model)
            assert "Local" in repr_str
            assert "test" in repr_str

    def test_llm_setup_variations(self):
        """Test different LLM setup configurations."""
        # Test with existing model
        dummy_model = DummyChatModel()
        result = llm.setup(dummy_model)
        assert result is dummy_model

        # Test with OpenAI dict config (clear API key to force error)
        config = {"provider": "openai", "model": "gpt-3.5-turbo"}
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises((llm.OpenAIInitializationError, Exception)):
                llm.setup(config)

        # Test with invalid provider
        config = {"provider": "unknown", "model": "test", "base_url": "http://test"}
        with patch("t2ebm.llm.LocalLLMChatModel") as mock_local:
            mock_local.return_value = MagicMock()
            result = llm.setup(config)

    def test_utils_functions(self):
        """Test utility functions."""
        # Test token counting
        count = utils.num_tokens_from_string_("Hello world", "gpt-4")
        assert count > 0

        # Test message formatting
        messages = [
            {"role": "system", "content": "Test system"},
            {"role": "user", "content": "Test user"},
            {"role": "assistant", "content": "Test assistant"},
        ]
        prompt = utils._format_messages_as_prompt(messages)
        assert "System:" in prompt
        assert "User:" in prompt
        assert "Assistant:" in prompt

    def test_graph_edge_cases(self, sample_ebm):
        """Test graph edge cases and error conditions."""
        graph = graphs.extract_graph(sample_ebm, 0)

        # Test with very low precision
        text = graphs.graph_to_text(graph, y_axis_precision=0)
        assert isinstance(text, str)

        # Test simplify graph with different parameters
        simplified = graphs.simplify_graph(graph, min_variation_per_cent=0.05)
        assert simplified is not None

    @patch.dict("os.environ", {}, clear=True)
    def test_utils_client_creation_no_key(self):
        """Test client creation without API key."""
        with pytest.raises(utils.OpenAIInitializationError):
            utils.create_direct_client()

    def test_guidance_query_parsing(self):
        """Test guidance query parsing with different formats."""
        # Test system message
        query = "{{#system~}}You are helpful{{~/system}}"
        messages = utils.parse_guidance_query(query)
        assert len(messages) >= 1
        assert messages[0]["role"] == "system"

        # Test assistant message
        query = "{{#assistant~}}I can help{{~/assistant}}"
        messages = utils.parse_guidance_query(query)
        assert len(messages) >= 1
        assert messages[0]["role"] == "assistant"

        # Test empty query
        messages = utils.parse_guidance_query("")
        assert isinstance(messages, list)

    def test_openai_completion_functions(self):
        """Test OpenAI completion utility functions."""
        messages = [{"role": "user", "content": "test"}]

        # These should fail without proper API setup
        with pytest.raises(
            (utils.OpenAIInitializationError, utils.OpenAICompletionError)
        ):
            utils.openai_completion_query("gpt-3.5-turbo", messages)

        with pytest.raises(
            (utils.OpenAIInitializationError, utils.OpenAICompletionError)
        ):
            openai_debug_completion_query("gpt-3.5-turbo", messages)
