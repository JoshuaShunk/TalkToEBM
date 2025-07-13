"""
Test LLM functionality and chat models.
"""

from unittest.mock import MagicMock, patch

import pytest

import t2ebm.llm as llm


class TestDummyChatModel:
    """Test the dummy chat model."""

    def test_dummy_chat_model_creation(self):
        """Test creating a dummy chat model."""
        model = llm.DummyChatModel()
        assert isinstance(model, llm.AbstractChatModel)

    def test_dummy_chat_completion(self):
        """Test dummy chat completion."""
        model = llm.DummyChatModel()
        messages = [{"role": "user", "content": "Test message"}]

        response = model.chat_completion(messages, temperature=0.7, max_tokens=100)
        assert isinstance(response, str)
        assert len(response) > 0


class TestLocalLLMChatModel:
    """Test local LLM chat model functionality."""

    def test_local_llm_invalid_base_url(self):
        """Test that invalid base URL raises error."""
        with pytest.raises(llm.LocalLLMError):
            llm.LocalLLMChatModel("", "test-model")

    @patch("requests.get")
    def test_local_llm_connection_test(self, mock_get):
        """Test connection testing."""
        mock_get.return_value.status_code = 200

        # Should not raise an exception
        model = llm.LocalLLMChatModel("http://localhost:11434", "test-model")
        assert model.base_url == "http://localhost:11434"
        assert model.model == "test-model"

    @patch("t2ebm.llm.LocalLLMChatModel._is_ollama")
    @patch("t2ebm.llm.LocalLLMChatModel._test_connection")
    def test_local_llm_connection_failure(self, mock_test, mock_is_ollama):
        """Test connection failure handling."""
        mock_is_ollama.return_value = False
        mock_test.side_effect = llm.LocalLLMError("Connection failed")

        with pytest.raises(llm.LocalLLMError):
            llm.LocalLLMChatModel("http://localhost:11434", "test-model")

    def test_deepseek_thinking_filter(self):
        """Test filtering of DeepSeek thinking sections."""
        model = llm.LocalLLMChatModel.__new__(llm.LocalLLMChatModel)

        text_with_thinking = "Some text <think>internal thoughts</think> more text"
        filtered = model._filter_deepseek_thinking(text_with_thinking)
        assert "<think>" not in filtered
        assert "</think>" not in filtered
        assert "Some text" in filtered
        assert "more text" in filtered


class TestOpenAIChatModel:
    """Test OpenAI chat model functionality."""

    def test_openai_model_invalid_client(self):
        """Test that invalid client raises error."""
        with pytest.raises(llm.OpenAIInitializationError):
            llm.OpenAIChatModel(None, "gpt-3.5-turbo")

    @patch("t2ebm.llm.OPENAI_V1", True)
    def test_openai_model_creation(self):
        """Test creating OpenAI model with mock client."""
        mock_client = MagicMock()
        model = llm.OpenAIChatModel(mock_client, "gpt-3.5-turbo")
        assert model.client == mock_client
        assert model.model == "gpt-3.5-turbo"


class TestSetupFunctions:
    """Test model setup functions."""

    def test_setup_with_abstract_model(self):
        """Test setup with an existing AbstractChatModel."""
        dummy_model = llm.DummyChatModel()
        result = llm.setup(dummy_model)
        assert result is dummy_model

    def test_setup_with_dict_local(self):
        """Test setup with dictionary configuration for local LLM."""
        config = {
            "provider": "local",
            "model": "test-model",
            "base_url": "http://localhost:11434",
        }

        with patch("t2ebm.llm.LocalLLMChatModel") as mock_local:
            mock_local.return_value = MagicMock()
            result = llm.setup(config)
            mock_local.assert_called_once_with("http://localhost:11434", "test-model")

    def test_setup_invalid_input(self):
        """Test setup with invalid input."""
        with pytest.raises(ValueError):
            llm.setup(123)  # Invalid type

    def test_setup_dict_missing_requirements(self):
        """Test setup with dictionary missing required fields."""
        config = {"provider": "local"}  # Missing model and base_url

        with pytest.raises(ValueError):
            llm.setup(config)


class TestChatCompletion:
    """Test chat completion functionality."""

    def test_chat_completion_basic(self):
        """Test basic chat completion."""
        dummy_model = llm.DummyChatModel()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "temperature": 0.7, "max_tokens": 100},
        ]

        result = llm.chat_completion(dummy_model, messages)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "content" in result[1]

    def test_chat_completion_with_existing_content(self):
        """Test chat completion with assistant message that already has content."""
        dummy_model = llm.DummyChatModel()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = llm.chat_completion(dummy_model, messages)
        assert result[1]["content"] == "Hi there!"
