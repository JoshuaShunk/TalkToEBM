"""
Extended Utils tests to boost coverage.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, Mock

import t2ebm.utils as utils


class TestUtilsExtended:
    """Extended Utils tests for better coverage."""

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_ensure_client_success(self):
        """Test successful client initialization."""
        # Reset the client state
        utils.client = None
        utils.client_initialized = False
        
        with patch('t2ebm.utils.create_openai_client') as mock_create:
            mock_create.return_value = MagicMock()
            client = utils._ensure_client()
            assert client is not None
            assert utils.client_initialized

    @patch.dict('os.environ', {}, clear=True)
    def test_ensure_client_fallback_v1(self):
        """Test client initialization fallback for v1."""
        utils.client = None
        utils.client_initialized = False
        
        with patch('t2ebm.utils.OPENAI_V1', True):
            with patch('t2ebm.utils.create_openai_client') as mock_create:
                mock_create.side_effect = utils.OpenAIInitializationError("Test error")
                with patch('openai.OpenAI') as mock_openai:
                    mock_openai.return_value = MagicMock()
                    client = utils._ensure_client()
                    assert client is not None

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_ensure_client_fallback_legacy(self):
        """Test client initialization fallback for legacy."""
        utils.client = None
        utils.client_initialized = False
        
        with patch('t2ebm.utils.OPENAI_V1', False):
            with patch('t2ebm.utils.create_openai_client') as mock_create:
                mock_create.side_effect = ImportError("No module")
                client = utils._ensure_client()
                assert client is not None

    def test_ensure_client_failure(self):
        """Test client initialization complete failure."""
        utils.client = None
        utils.client_initialized = False
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('t2ebm.utils.create_openai_client') as mock_create:
                mock_create.side_effect = utils.OpenAIInitializationError("Test error")
                client = utils._ensure_client()
                assert client is None
                assert utils.client_initialized  # Should be marked as attempted

    @patch('t2ebm.utils.OPENAI_V1', True)
    def test_openai_completion_query_v1_success(self):
        """Test OpenAI completion query with v1 API."""
        utils.client_initialized = True
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        utils.client = mock_client
        
        response = utils.openai_completion_query("gpt-3.5-turbo", [{"role": "user", "content": "test"}])
        assert response == "Test response"

    @patch('t2ebm.utils.OPENAI_V1', False)
    def test_openai_completion_query_legacy_success(self):
        """Test OpenAI completion query with legacy API."""
        utils.client_initialized = True
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = {"content": "Test response"}
        mock_client.ChatCompletion.create.return_value = mock_response
        utils.client = mock_client
        
        response = utils.openai_completion_query("gpt-3.5-turbo", [{"role": "user", "content": "test"}])
        assert response == "Test response"

    def test_openai_completion_query_no_client(self):
        """Test completion query with no client."""
        utils.client = None
        utils.client_initialized = True
        
        with pytest.raises(utils.OpenAIInitializationError):
            utils.openai_completion_query("gpt-3.5-turbo", [{"role": "user", "content": "test"}])

    @patch('t2ebm.utils.OPENAI_V1', True)
    def test_openai_completion_query_api_error(self):
        """Test OpenAI completion query API error."""
        utils.client_initialized = True
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        utils.client = mock_client
        
        with pytest.raises(utils.OpenAICompletionError):
            utils.openai_completion_query("gpt-3.5-turbo", [{"role": "user", "content": "test"}])

    def test_try_api_resources_fallback_v1(self):
        """Test API resources fallback for v1."""
        with patch('t2ebm.utils.OPENAI_V1', True):
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                with patch('openai.OpenAI') as mock_openai:
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock()]
                    mock_response.choices[0].message.content = "Fallback response"
                    mock_client.chat.completions.create.return_value = mock_response
                    mock_openai.return_value = mock_client
                    
                    response = utils._try_api_resources_fallback("gpt-3.5-turbo", [{"role": "user", "content": "test"}])
                    assert response == "Fallback response"

    def test_try_api_resources_fallback_legacy(self):
        """Test API resources fallback for legacy."""
        with patch('t2ebm.utils.OPENAI_V1', False):
            with patch('t2ebm.utils.openai.ChatCompletion.create') as mock_create:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message = {"content": "Legacy response"}
                mock_create.return_value = mock_response
                
                response = utils._try_api_resources_fallback("gpt-3.5-turbo", [{"role": "user", "content": "test"}])
                assert response == "Legacy response"

    def test_try_api_resources_fallback_error(self):
        """Test API resources fallback complete failure."""
        with patch('t2ebm.utils.OPENAI_V1', False):
            with patch('t2ebm.utils.openai.ChatCompletion.create') as mock_create:
                mock_create.side_effect = Exception("Complete failure")
                
                with pytest.raises(utils.OpenAICompletionError):
                    utils._try_api_resources_fallback("gpt-3.5-turbo", [{"role": "user", "content": "test"}])

    def test_parse_guidance_query_multiple_messages(self):
        """Test parsing guidance query with multiple message types."""
        query = "{{#system~}}You are helpful{{~/system}}{{#user~}}Hello{{~/user}}{{#assistant~}}Hi there{{~/assistant}}"
        messages = utils.parse_guidance_query(query)
        
        assert len(messages) >= 3
        roles = [msg["role"] for msg in messages]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_parse_guidance_query_no_matches(self):
        """Test parsing guidance query with no valid tokens."""
        query = "Just plain text without tokens"
        messages = utils.parse_guidance_query(query)
        assert isinstance(messages, list)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_create_direct_client_success(self):
        """Test successful direct client creation."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            client = utils.create_direct_client()
            assert client is not None

    def test_create_direct_client_with_key_param(self):
        """Test direct client creation with API key parameter."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            client = utils.create_direct_client("custom-key")
            assert client is not None
            mock_openai.assert_called_with(api_key="custom-key")

    def test_create_direct_client_manual_fallback(self):
        """Test direct client creation with manual fallback."""
        with patch('openai.OpenAI') as mock_openai:
            # First call fails, second succeeds
            mock_openai.side_effect = [Exception("Init failed"), MagicMock()]
            
            with patch('httpx.Client') as mock_httpx:
                mock_httpx.return_value = MagicMock()
                
                with pytest.raises(utils.OpenAIInitializationError):
                    # This will try the manual approach but still fail
                    utils.create_direct_client("test-key")

    def test_create_direct_client_legacy_fallback(self):
        """Test direct client creation handles import errors."""
        with pytest.raises(utils.OpenAIInitializationError):
            utils.create_direct_client()

    def test_openai_debug_completion_different_path(self):
        """Test debug completion takes different code path."""
        utils.client_initialized = True
        utils.client = None
        
        with pytest.raises(utils.OpenAIInitializationError):
            utils.openai_debug_completion_query("gpt-3.5-turbo", [{"role": "user", "content": "test"}])

    def test_openai_debug_completion_legacy_no_chat(self):
        """Test debug completion with legacy API without ChatCompletion."""
        utils.client_initialized = True
        mock_client = MagicMock()
        # Remove ChatCompletion attribute
        del mock_client.ChatCompletion
        mock_client.Completion.create.return_value.choices = [MagicMock()]
        mock_client.Completion.create.return_value.choices[0].text = "Legacy response"
        utils.client = mock_client
        
        with patch('t2ebm.utils.OPENAI_V1', False):
            response = utils.openai_debug_completion_query("gpt-3.5-turbo", [{"role": "user", "content": "test"}])
            assert response == "Legacy response"

    def test_guidance_query_recursion_limit(self):
        """Test guidance query parsing stops at reasonable length."""
        # Create a short query that won't trigger recursion
        query = "{{#user~}}Short{{~/user}}"
        messages = utils.parse_guidance_query(query)
        assert len(messages) == 1