"""
TalkToEBM structures conversations in a generic OpenAI message format that can be executed with different LLMs.

We interface the LLM via the simple class AbstractChatModel. To use your own LLM, simply implement the chat_completion method in a subclass.
"""

from dataclasses import dataclass
import copy
import os
import sys
import json
import requests
from typing import Union, List, Dict, Any, Optional

# Global flag to track if we've already tried and failed to initialize OpenAI
OPENAI_INIT_ATTEMPTED = False
OPENAI_INIT_FAILED = False

# Detect OpenAI API version
try:
    # For OpenAI v1.0+
    from openai import OpenAI, AzureOpenAI
    OPENAI_V1 = True
except ImportError:
    # For OpenAI <v1.0
    import openai
    OPENAI_V1 = False


class OpenAIInitializationError(Exception):
    """Exception raised when OpenAI client initialization fails."""
    pass


class OpenAICompletionError(Exception):
    """Exception raised when OpenAI completion fails."""
    pass


class LocalLLMError(Exception):
    """Exception raised when a local LLM request fails."""
    pass


@dataclass
class AbstractChatModel:
    """
    Abstract base class for chat models. Implement chat_completion to create a custom model.
    """
    def chat_completion(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> str:
        """Send a query to a chat model.

        :param messages: The messages to send to the model. We use the OpenAI format.
        :param temperature: The sampling temperature.
        :param max_tokens: The maximum number of tokens to generate.

        Returns:
            str: The model response.
        """
        raise NotImplementedError


class LocalLLMChatModel(AbstractChatModel):
    """
    Chat model that uses a local LLM API (like Ollama) via HTTP.
    Compatible with Ollama and other APIs that use a similar format.
    """
    
    def __init__(self, base_url: str, model: str):
        """
        Initialize with a base URL and model name.
        
        Args:
            base_url: The base URL for the LLM API (e.g., 'http://localhost:11434')
            model: The name of the model to use (e.g., 'llama2')
        """
        super().__init__()
        if not base_url:
            raise LocalLLMError("Base URL for local LLM is required")
        
        # Remove trailing slash if present
        if base_url.endswith('/'):
            base_url = base_url[:-1]
            
        self.base_url = base_url
        self.model = model
        
        # Detect API type (Ollama or other)
        if "ollama" in base_url.lower() or self._is_ollama():
            self.api_type = "ollama"
        else:
            self.api_type = "generic"
            
        # Test the connection
        self._test_connection()
    
    def _is_ollama(self) -> bool:
        """
        Test if the API is Ollama by checking for Ollama-specific endpoints.
        
        Returns:
            bool: True if Ollama API, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _test_connection(self) -> None:
        """
        Test the connection to the LLM API.
        
        Raises:
            LocalLLMError: If the connection test fails
        """
        try:
            if self.api_type == "ollama":
                # Test with a simple model check for Ollama
                url = f"{self.base_url}/api/tags"
                response = requests.get(url, timeout=5)
                if response.status_code != 200:
                    raise LocalLLMError(f"Failed to connect to Ollama API at {self.base_url}: HTTP {response.status_code}")
            else:
                # Generic test - just make sure we can reach the server
                url = self.base_url
                response = requests.get(url, timeout=5)
                if response.status_code >= 500:
                    raise LocalLLMError(f"Failed to connect to LLM API at {self.base_url}: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise LocalLLMError(f"Failed to connect to LLM API at {self.base_url}: {str(e)}")
    
    def chat_completion(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> str:
        """
        Send a chat completion request to the local LLM API.
        
        Args:
            messages: The messages to send to the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: The model's response
            
        Raises:
            LocalLLMError: If the completion request fails
        """
        try:
            if self.api_type == "ollama":
                return self._ollama_completion(messages, temperature, max_tokens)
            else:
                return self._generic_completion(messages, temperature, max_tokens)
        except Exception as e:
            raise LocalLLMError(f"Failed to complete request: {str(e)}")
    
    def _filter_deepseek_thinking(self, response_text: str) -> str:
        """
        Filter out thinking sections from DeepSeek model responses.
        
        DeepSeek models sometimes include <think>...</think> sections in their responses
        that should be filtered out before returning to the user.
        
        Args:
            response_text: The raw response from the DeepSeek model
            
        Returns:
            str: The filtered response with thinking sections removed
        """
        if not response_text:
            return response_text
            
        # Remove sections enclosed in <think>...</think> tags
        import re
        filtered_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        
        # Also handle variations like `think>...` without brackets
        filtered_text = re.sub(r'think>.*?</think>', '', filtered_text, flags=re.DOTALL)
        
        # Clean up any leftover empty lines from the removal
        filtered_text = re.sub(r'\n\s*\n', '\n\n', filtered_text)
        
        return filtered_text.strip()
    
    def _ollama_completion(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> str:
        """
        Send a completion request to an Ollama API.
        
        Args:
            messages: The messages to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: The model's response
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Some models might require additional parameters or format adjustments
        if any(model_type in self.model.lower() for model_type in ['deepseek', 'mistral', 'mixtral', 'llama', 'yi', 'phi', 'gemma']):
            # Add any model-specific parameters if needed
            # Currently Ollama handles most models with the same API format
            pass
        
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code != 200:
            raise LocalLLMError(f"Ollama API returned error status: {response.status_code} - {response.text}")
        
        response_data = response.json()
        
        # Try different response formats - different models might return different structures
        try:
            content = None
            
            # Most common format
            if "message" in response_data and "content" in response_data["message"]:
                content = response_data["message"]["content"]
            # Alternative format sometimes used
            elif "response" in response_data:
                content = response_data["response"]
            # Some models might return this format
            elif "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    content = response_data["choices"][0]["message"]["content"]
                elif "text" in response_data["choices"][0]:
                    content = response_data["choices"][0]["text"]
            # DeepSeek specific format (if needed)
            elif "deepseek" in self.model.lower() and "output" in response_data:
                content = response_data["output"]
                
            # If we got content and this is a DeepSeek model, filter out thinking sections
            if content and 'deepseek' in self.model.lower():
                return self._filter_deepseek_thinking(content)
                
            # Return the content if we found it
            if content:
                return content
                
        except (KeyError, TypeError, IndexError):
            pass
        
        # If we couldn't parse the response using the expected formats, raise an error with the raw response
        raise LocalLLMError(f"Unexpected response format from Ollama API: {response_data}")
    
    def _generic_completion(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> str:
        """
        Send a completion request to a generic API using OpenAI-like format.
        
        Args:
            messages: The messages to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: The model's response
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        # Use OpenAI-compatible format
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code != 200:
            raise LocalLLMError(f"LLM API returned error status: {response.status_code} - {response.text}")
        
        response_data = response.json()
        
        try:
            # Extract in OpenAI-compatible format
            return response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LocalLLMError(f"Unexpected response format from LLM API: {response_data}")

    def __repr__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            str: Information about the model and API
        """
        return f"Local {self.api_type.capitalize()} LLM: {self.model} @ {self.base_url}"


class OpenAIChatModel(AbstractChatModel):
    """
    Chat model that uses the OpenAI API.
    Compatible with both v1.0+ and older versions of the API.
    """
    client = None
    model: str = None

    def __init__(self, client, model: str):
        """
        Initialize with an OpenAI client and model name.
        
        Args:
            client: OpenAI client (v1.0+ or older version)
            model: The model name to use
        """
        super().__init__()
        if client is None:
            raise OpenAIInitializationError("OpenAI client is not initialized. Check your API key and environment.")
        self.client = client
        self.model = model

    def chat_completion(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> str:
        """
        Send a chat completion request to the OpenAI API.
        
        Args:
            messages: The messages to send to the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: The model's response
            
        Raises:
            OpenAICompletionError: If the completion request fails
        """
        global OPENAI_INIT_FAILED
        
        # If we know OpenAI initialization has failed, raise exception
        if OPENAI_INIT_FAILED:
            raise OpenAIInitializationError("OpenAI initialization previously failed.")
            
        try:
            # Handle OpenAI v1.0+ API
            if OPENAI_V1:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=90,
                )
                # Extract the completion text
                response_content = getattr(response.choices[0].message, "content", "")
                if response_content is None:
                    response_content = ""
                return response_content
                
            # Handle older OpenAI API versions
            if hasattr(self.client, "ChatCompletion"):
                response = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=90,
                )
                # Extract content based on structure
                if hasattr(response.choices[0], "message"):
                    return response.choices[0].message["content"] or ""
                return response.choices[0]["message"]["content"] or ""
                
            # For very old versions that might use a different structure
            prompt = self._format_messages_as_prompt(messages)
            response = self.client.Completion.create(
                engine=self.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=90,
            )
            return response.choices[0].text.strip()
            
        except Exception as e:
            # Raise an exception with the error details
            raise OpenAICompletionError(f"Failed to complete request: {str(e)}")
    
    def _format_messages_as_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert chat messages to a single prompt string for older API versions.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: Formatted prompt string
        """
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
        return prompt

    def __repr__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            str: The model name
        """
        return f"OpenAI: {self.model}"


def create_openai_client(azure: bool = False) -> Optional[Any]:
    """
    Create an OpenAI client with proper error handling.
    
    Args:
        azure: Whether to create an Azure OpenAI client
        
    Returns:
        The OpenAI client
        
    Raises:
        OpenAIInitializationError: If client initialization fails
    """
    global OPENAI_INIT_ATTEMPTED, OPENAI_INIT_FAILED
    
    # If we've already tried and failed, don't try again
    if OPENAI_INIT_ATTEMPTED and OPENAI_INIT_FAILED:
        raise OpenAIInitializationError("OpenAI client initialization has already failed.")
        
    OPENAI_INIT_ATTEMPTED = True
    
    # Silently check for proxy settings in environment but don't print anything
    proxy_vars = [var for var in os.environ if "proxy" in var.lower()]
    
    if not OPENAI_V1:
        # For older versions, just return the openai module
        return openai
        
    # For OpenAI v1.0+
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not azure:
        OPENAI_INIT_FAILED = True
        raise OpenAIInitializationError("OPENAI_API_KEY environment variable is not set.")
    
    # Try directly creating the client with no parameters first
    try:
        if azure:
            from openai import AzureOpenAI
            return AzureOpenAI()
        else:
            from openai import OpenAI
            return OpenAI()
    except Exception:
        pass
    
    # Try with absolutely minimal parameters - just the API key
    try:
        if azure:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            azure_key = os.environ.get("AZURE_OPENAI_KEY")
            if not azure_endpoint or not azure_key:
                OPENAI_INIT_FAILED = True
                raise OpenAIInitializationError("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_KEY environment variables are not set.")
                
            from openai import AzureOpenAI
            return AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key
            )
        else:
            from openai import OpenAI
            # Try with a custom __init__ approach to bypass potential issues
            try:
                # Create object then set attributes directly to avoid __init__ issues
                client = object.__new__(OpenAI)
                
                # Manually set attributes to bypass __init__
                import httpx
                http_client = httpx.Client()
                setattr(client, "_client", http_client)
                setattr(client, "api_key", api_key)
                
                # Set required attributes
                return client
            except Exception:
                # Try normal initialization as fallback
                return OpenAI(api_key=api_key)
    except Exception as e:
        OPENAI_INIT_FAILED = True
        raise OpenAIInitializationError(f"Failed to initialize OpenAI client: {str(e)}")


def openai_setup(model: str, azure: bool = False, *args, **kwargs) -> AbstractChatModel:
    """
    Setup an OpenAI language model.

    Args:
        model: The name of the model (e.g. "gpt-3.5-turbo-0613").
        azure: If true, use a model deployed on azure.

    This function uses the following environment variables:
    - OPENAI_API_KEY
    - OPENAI_API_ORG
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_KEY
    - AZURE_OPENAI_VERSION

    Returns:
        AbstractChatModel: An OpenAI chat model
        
    Raises:
        OpenAIInitializationError: If initialization fails
    """
    # Create a client that works with the current OpenAI version
    client = create_openai_client(azure)
    
    # If we're using the legacy API, set up the environment variables
    if not OPENAI_V1 and client:
        if "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        elif not azure:
            raise OpenAIInitializationError("OPENAI_API_KEY environment variable is not set.")
            
        if "OPENAI_API_ORG" in os.environ:
            openai.organization = os.environ["OPENAI_API_ORG"]
        
        if azure:
            openai.api_type = "azure"
            if "AZURE_OPENAI_ENDPOINT" in os.environ:
                openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
            else:
                raise OpenAIInitializationError("AZURE_OPENAI_ENDPOINT environment variable is not set.")
                
            if "AZURE_OPENAI_KEY" in os.environ:
                openai.api_key = os.environ["AZURE_OPENAI_KEY"]
            else:
                raise OpenAIInitializationError("AZURE_OPENAI_KEY environment variable is not set.")
                
            if "AZURE_OPENAI_VERSION" in os.environ:
                openai.api_version = os.environ["AZURE_OPENAI_VERSION"]

    # Create the model
    return OpenAIChatModel(client, model)


def local_llm_setup(model: str, base_url: str) -> AbstractChatModel:
    """
    Setup a local LLM (like Ollama).
    
    Args:
        model: The name of the model to use (e.g., "llama2")
        base_url: The base URL for the LLM API (e.g., "http://localhost:11434")
        
    Returns:
        AbstractChatModel: A local LLM chat model
        
    Raises:
        LocalLLMError: If initialization fails
    """
    return LocalLLMChatModel(base_url, model)


def setup(model: Union[AbstractChatModel, str, Dict[str, Any]]) -> AbstractChatModel:
    """
    Setup a chat model. Supports multiple ways to specify the model:
    
    1. If the input is an AbstractChatModel instance, return it directly.
    2. If the input is a string, assume it's an OpenAI model name.
    3. If the input is a dict with "provider", "model", and other settings, use those.
    
    Args:
        model: Either an AbstractChatModel instance, a string (OpenAI model name),
              or a dict with provider/model/settings info
        
    Returns:
        AbstractChatModel: The chat model
        
    Raises:
        OpenAIInitializationError: If OpenAI model setup fails
        LocalLLMError: If local LLM setup fails
        ValueError: If the model specification is invalid
    """
    # If already a model instance, just return it
    if isinstance(model, AbstractChatModel):
        return model
        
    # If a string, assume it's an OpenAI model name
    if isinstance(model, str):
        return openai_setup(model)
        
    # If a dict, check the provider and set up accordingly
    if isinstance(model, dict):
        provider = model.get("provider", "openai").lower()
        
        if provider == "openai":
            # Set up OpenAI client
            model_name = model.get("model", "gpt-3.5-turbo")
            azure = model.get("azure", False)
            return openai_setup(model_name, azure)
            
        elif provider in ["local", "ollama", "deepseek", "llama", "mistral", "mixtral", "gemma", "yi", "phi"]:
            # Set up local LLM - support various model types running through Ollama or other local API servers
            model_name = model.get("model", "llama2")
            base_url = model.get("base_url")
            if not base_url:
                raise ValueError("base_url is required for local LLM setup")
            return local_llm_setup(model_name, base_url)
            
        else:
            # Assume any other provider is a model running on a local server
            # This is much more flexible and allows users to specify any model type
            model_name = model.get("model")
            base_url = model.get("base_url")
            
            if not model_name:
                raise ValueError(f"Model name is required for provider: {provider}")
            if not base_url:
                raise ValueError(f"base_url is required for provider: {provider}")
                
            return local_llm_setup(model_name, base_url)
    
    # If we got here, the input format is invalid
    raise ValueError(
        "Invalid model specification. Must be either: "
        "1. An AbstractChatModel instance, "
        "2. A string (OpenAI model name), or "
        "3. A dict with 'provider', 'model', and other settings."
    )


def chat_completion(llm: Union[str, AbstractChatModel, Dict[str, Any]], messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Execute a sequence of user and assistant messages with an AbstractChatModel.

    Sends multiple individual messages to the AbstractChatModel.
    
    Args:
        llm: The language model to use (string, AbstractChatModel instance, or dict)
        messages: The messages to process
        
    Returns:
        List[Dict[str, Any]]: The processed messages
        
    Raises:
        OpenAIInitializationError: If model initialization fails
        OpenAICompletionError: If a completion request fails
        LocalLLMError: If a local LLM request fails
    """
    llm = setup(llm)
    # Make a copy to avoid modifying the input
    messages = copy.deepcopy(messages)
    
    # Process assistant messages that don't have content
    for msg_idx in range(len(messages)):
        if messages[msg_idx]["role"] == "assistant":
            if "content" not in messages[msg_idx]:
                # Get required parameters
                temperature = messages[msg_idx].get("temperature", 0.7)
                max_tokens = messages[msg_idx].get("max_tokens", 1000)
                
                # Generate content
                messages[msg_idx]["content"] = llm.chat_completion(
                    messages[:msg_idx],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            
            # Remove all keys except "role" and "content"
            messages[msg_idx] = {
                "role": messages[msg_idx]["role"],
                "content": messages[msg_idx]["content"]
            }
            
    return messages
