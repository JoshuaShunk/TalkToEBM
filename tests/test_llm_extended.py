"""
Extended LLM tests to boost coverage.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

import t2ebm.llm as llm
from t2ebm.testing import DummyChatModel


class TestLLMExtended:
    """Extended LLM tests for better coverage."""

    def test_openai_v1_detection(self):
        """Test OpenAI v1 detection and imports."""
        # Test that the module properly detects OpenAI version
        assert isinstance(llm.OPENAI_V1, bool)

    @patch.dict("os.environ", {}, clear=True)
    def test_create_openai_client_success(self):
        """Test that create_openai_client handles missing API keys."""
        with pytest.raises(llm.OpenAIInitializationError):
            llm.create_openai_client()

    @patch.dict("os.environ", {}, clear=True)
    @patch("t2ebm.llm.OPENAI_V1", True)
    def test_create_openai_client_no_key(self):
        """Test OpenAI client creation without API key."""
        with pytest.raises(llm.OpenAIInitializationError):
            llm.create_openai_client()

    def test_openai_setup_legacy(self):
        """Test OpenAI setup without proper credentials."""
        with pytest.raises(llm.OpenAIInitializationError):
            llm.openai_setup("gpt-3.5-turbo")

    def test_openai_setup_azure_legacy(self):
        """Test Azure OpenAI setup without proper credentials."""
        with pytest.raises(llm.OpenAIInitializationError):
            llm.openai_setup("gpt-3.5-turbo", azure=True)

    def test_local_llm_base_url_cleanup(self):
        """Test that base URL trailing slash is removed."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            model = llm.LocalLLMChatModel("http://localhost:11434/", "test")
            assert model.base_url == "http://localhost:11434"

    @patch("requests.get")
    def test_local_llm_generic_api_detection(self, mock_get):
        """Test generic API detection (non-Ollama)."""
        # First call (is_ollama check) returns 404, second call (test_connection) returns 200
        mock_get.side_effect = [Mock(status_code=404), Mock(status_code=200)]

        model = llm.LocalLLMChatModel("http://localhost:8000", "test")
        assert model.api_type == "generic"

    @patch("requests.post")
    @patch("requests.get")
    def test_local_llm_ollama_completion(self, mock_get, mock_post):
        """Test Ollama completion."""
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {"content": "Test response"}
        }

        model = llm.LocalLLMChatModel("http://localhost:11434", "llama2")
        response = model.chat_completion(
            [{"role": "user", "content": "test"}], 0.7, 100
        )
        assert response == "Test response"

    @patch("requests.post")
    @patch("requests.get")
    def test_local_llm_generic_completion(self, mock_get, mock_post):
        """Test generic API completion."""
        # is_ollama returns 404, test_connection returns 200
        mock_get.side_effect = [Mock(status_code=404), Mock(status_code=200)]
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }

        model = llm.LocalLLMChatModel("http://localhost:8000", "test-model")
        response = model.chat_completion(
            [{"role": "user", "content": "test"}], 0.7, 100
        )
        assert response == "Test response"

    @patch("requests.post")
    @patch("requests.get")
    def test_local_llm_deepseek_filtering(self, mock_get, mock_post):
        """Test DeepSeek thinking section filtering."""
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {"content": "Some text <think>thinking</think> more text"}
        }

        model = llm.LocalLLMChatModel("http://localhost:11434", "deepseek-coder")
        response = model.chat_completion(
            [{"role": "user", "content": "test"}], 0.7, 100
        )
        assert "<think>" not in response
        assert "Some text" in response
        assert "more text" in response

    @patch("requests.post")
    @patch("requests.get")
    def test_local_llm_api_error(self, mock_get, mock_post):
        """Test API error handling."""
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Server Error"

        model = llm.LocalLLMChatModel("http://localhost:11434", "test")
        with pytest.raises(llm.LocalLLMError):
            model.chat_completion([{"role": "user", "content": "test"}], 0.7, 100)

    def test_openai_model_with_mock_client(self):
        """Test OpenAI model with a mock client."""
        mock_client = MagicMock()
        model = llm.OpenAIChatModel(mock_client, "gpt-3.5-turbo")
        assert model.client == mock_client
        assert model.model == "gpt-3.5-turbo"

    def test_setup_invalid_dict_config(self):
        """Test setup with invalid dictionary configurations."""
        # Missing base_url for local provider
        config = {"provider": "local", "model": "test"}
        with pytest.raises(ValueError):
            llm.setup(config)

        # Missing model for unknown provider
        config = {"provider": "custom"}
        with pytest.raises(ValueError):
            llm.setup(config)

    def test_chat_completion_message_processing(self):
        """Test chat completion message processing logic."""
        dummy_model = DummyChatModel()
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "temperature": 0.5,
                "max_tokens": 50,
                "extra_key": "value",
            },
        ]

        result = llm.chat_completion(dummy_model, messages)
        # Should clean extra keys from assistant message
        assert "extra_key" not in result[1]
        assert result[1]["role"] == "assistant"
        assert "content" in result[1]
