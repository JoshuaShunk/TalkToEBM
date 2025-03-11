"""
TalkToEBM structures conversations in a generic OpenAI message format that can be executed with different LLMs.

We interface the LLM via the simple class AbstractChatModel. To use your own LLM, simply implement the chat_completion method in a subclass.
"""

from dataclasses import dataclass
import copy
import os
import sys
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


class DummyChatModel(AbstractChatModel):
    """
    A dummy chat model for testing or fallback when real models aren't available.
    """
    def chat_completion(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> str:
        """Return a dummy response regardless of input."""
        return "Hihi, this is the dummy chat model! I hope you find me useful for debugging."


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
        """
        global OPENAI_INIT_FAILED
        
        # If we know OpenAI initialization has failed, use DummyChatModel instead
        if OPENAI_INIT_FAILED:
            dummy = DummyChatModel()
            return dummy.chat_completion(messages, temperature, max_tokens)
            
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
            
        except Exception:
            # Fall back to dummy model
            dummy = DummyChatModel()
            return dummy.chat_completion(messages, temperature, max_tokens)
    
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
        return f"{self.model}"


def create_openai_client(azure: bool = False) -> Optional[Any]:
    """
    Create an OpenAI client with proper error handling and fallbacks.
    
    Args:
        azure: Whether to create an Azure OpenAI client
        
    Returns:
        The OpenAI client or None if initialization failed
    """
    global OPENAI_INIT_ATTEMPTED, OPENAI_INIT_FAILED
    
    # If we've already tried and failed, don't try again
    if OPENAI_INIT_ATTEMPTED and OPENAI_INIT_FAILED:
        return None
        
    OPENAI_INIT_ATTEMPTED = True
    
    # Silently check for proxy settings in environment but don't print anything
    proxy_vars = [var for var in os.environ if "proxy" in var.lower()]
    
    if not OPENAI_V1:
        # For older versions, just return the openai module
        return openai
        
    # For OpenAI v1.0+
    api_key = os.environ.get("OPENAI_API_KEY")
    
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
                return None
                
            from openai import AzureOpenAI
            return AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key
            )
        else:
            if not api_key:
                OPENAI_INIT_FAILED = True
                return None
                
            from openai import OpenAI
            # Try with a custom __init__ approach to bypass potential issues
            try:
                # Create object then set attributes directly to avoid __init__ issues
                client = object.__new__(OpenAI)
                
                # Dangerous but might work - manually set attributes to bypass __init__
                import httpx
                http_client = httpx.Client()
                setattr(client, "_client", http_client)
                setattr(client, "api_key", api_key)
                
                # Set required attributes
                return client
            except Exception:
                # Try normal initialization as fallback
                return OpenAI(api_key=api_key)
    except Exception:
        OPENAI_INIT_FAILED = True
        return None


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
        AbstractChatModel: An LLM to work with
    """
    # Create a client that works with the current OpenAI version
    client = create_openai_client(azure)
    
    # If client creation failed, use DummyChatModel
    if not client and OPENAI_V1:
        return DummyChatModel()
        
    # If we're using the legacy API, set up the environment variables
    if not OPENAI_V1 and client:
        if "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        if "OPENAI_API_ORG" in os.environ:
            openai.organization = os.environ["OPENAI_API_ORG"]
        
        if azure:
            openai.api_type = "azure"
            if "AZURE_OPENAI_ENDPOINT" in os.environ:
                openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
            if "AZURE_OPENAI_KEY" in os.environ:
                openai.api_key = os.environ["AZURE_OPENAI_KEY"]
            if "AZURE_OPENAI_VERSION" in os.environ:
                openai.api_version = os.environ["AZURE_OPENAI_VERSION"]

    # Create the model
    return OpenAIChatModel(client, model)


def setup(model: Union[AbstractChatModel, str]) -> AbstractChatModel:
    """
    Setup a chat model. If the input is a string, we assume that it is the name of an OpenAI model.
    
    Args:
        model: Either an AbstractChatModel instance or a model name string
        
    Returns:
        AbstractChatModel: The chat model
    """
    if isinstance(model, str):
        model = openai_setup(model)
    return model


def chat_completion(llm: Union[str, AbstractChatModel], messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Execute a sequence of user and assistant messages with an AbstractChatModel.

    Sends multiple individual messages to the AbstractChatModel.
    
    Args:
        llm: The language model to use
        messages: The messages to process
        
    Returns:
        List[Dict[str, Any]]: The processed messages
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
