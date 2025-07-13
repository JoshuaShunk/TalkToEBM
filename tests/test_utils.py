"""
Test utility functions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from interpret.glassbox import ExplainableBoostingClassifier

import t2ebm.utils as utils


class TestUtils:
    """Test utility functions."""

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

    def test_num_tokens_from_string(self):
        """Test token counting function."""
        text = "This is a test sentence."
        count = utils.num_tokens_from_string_(text, "gpt-4")
        assert isinstance(count, int)
        assert count > 0

    def test_num_tokens_empty_string(self):
        """Test token counting with empty string."""
        count = utils.num_tokens_from_string_("", "gpt-4")
        assert count == 0

    def test_format_messages_as_prompt(self):
        """Test message formatting for older API versions."""
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        prompt = utils._format_messages_as_prompt(messages)
        assert isinstance(prompt, str)
        assert "System:" in prompt
        assert "User:" in prompt
        assert "Assistant:" in prompt

    def test_parse_guidance_query_basic(self):
        """Test basic guidance query parsing."""
        query = "{{#user~}}Hello{{~/user}}"
        messages = utils.parse_guidance_query(query)
        assert isinstance(messages, list)
        assert len(messages) > 0
        assert messages[0]["role"] == "user"
        assert "Hello" in messages[0]["content"]

    def test_create_direct_client_no_api_key(self):
        """Test direct client creation without API key."""
        with pytest.raises(utils.OpenAIInitializationError):
            utils.create_direct_client()
