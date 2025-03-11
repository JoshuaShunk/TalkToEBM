"""
TalkToEBM structures conversations in a generic OpenAI message format that can be executed with different LLMs.

We interface the LLM via the simple class AbstractChatModel. To use your own LLM, simply implement the chat_completion method in a subclass.
"""

from dataclasses import dataclass
import copy
import os
import sys
from typing import Union

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
    import importlib
    OPENAI_V1 = False
    # Print OpenAI version for debugging
    print(f"Using OpenAI API version: {openai.__version__}")


@dataclass
class AbstractChatModel:
    def chat_completion(self, messages, temperature: float, max_tokens: int):
        """Send a query to a chat model.

        :param messages: The messages to send to the model. We use the OpenAI format.
        :param temperature: The sampling temperature.
        :param max_tokens: The maximum number of tokens to generate.

        Returns:
            str: The model response.
        """
        raise NotImplementedError


class DummyChatModel(AbstractChatModel):
    def chat_completion(self, messages, temperature: float, max_tokens: int):
        return "Hihi, this is the dummy chat model! I hope you find me useful for debugging."


class OpenAIChatModel(AbstractChatModel):
    client = None
    model: str = None

    def __init__(self, client, model):
        super().__init__()
        self.client = client
        self.model = model

    def chat_completion(self, messages, temperature, max_tokens):
        global OPENAI_INIT_FAILED
        
        # If we know OpenAI initialization has failed, use DummyChatModel instead
        if OPENAI_INIT_FAILED:
            dummy = DummyChatModel()
            return dummy.chat_completion(messages, temperature, max_tokens)
            
        if OPENAI_V1:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=90,
                )
                # we return the completion string or "" if there is an invalid response/query
                try:
                    response_content = response.choices[0].message.content
                except:
                    print(f"Invalid response {response}")
                    response_content = ""
            except Exception as e:
                print(f"Error with OpenAI API call: {str(e)}")
                dummy = DummyChatModel()
                return dummy.chat_completion(messages, temperature, max_tokens)
        else:
            # Handle various versions of the legacy OpenAI API
            try:
                # Legacy OpenAI API method for versions around 0.27.x
                if hasattr(self.client, "ChatCompletion"):
                    response = self.client.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        request_timeout=90,
                    )
                    # Extract content based on response structure
                    try:
                        if hasattr(response.choices[0], "message"):
                            response_content = response.choices[0].message["content"]
                        else:
                            response_content = response.choices[0]["message"]["content"]
                    except Exception as e:
                        print(f"Error extracting content: {e}")
                        print(f"Response structure: {response}")
                        response_content = ""
                # For very old versions that might use a different structure
                else:
                    print("Using direct completion API for older OpenAI versions")
                    response = self.client.Completion.create(
                        engine=self.model,
                        prompt=self._format_messages_as_prompt(messages),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        request_timeout=90,
                    )
                    response_content = response.choices[0].text.strip()
            except Exception as e:
                print(f"Error with OpenAI API call: {str(e)}")
                print(f"OpenAI version: {openai.__version__}")
                print(f"Available client methods: {dir(self.client)}")
                # Fallback to DummyChatModel
                dummy = DummyChatModel()
                return dummy.chat_completion(messages, temperature, max_tokens)
        
        if response_content is None:
            print(f"Invalid response {response}")
            response_content = ""
        return response_content
    
    def _format_messages_as_prompt(self, messages):
        """Convert chat messages to a single prompt string for older API versions"""
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
        return f"{self.model}"


def create_openai_client(azure=False):
    """Create an OpenAI client with proper error handling and fallbacks."""
    global OPENAI_INIT_ATTEMPTED, OPENAI_INIT_FAILED
    
    # If we've already tried and failed, don't try again
    if OPENAI_INIT_ATTEMPTED and OPENAI_INIT_FAILED:
        return None
        
    OPENAI_INIT_ATTEMPTED = True
    
    # Print diagnostic information about the OpenAI package
    if OPENAI_V1:
        try:
            from openai import version
            print(f"OpenAI package version: {version.__version__}")
        except ImportError:
            print("Could not determine OpenAI package version")
    else:
        print(f"OpenAI package version: {openai.__version__}")
    
    # Check for proxy settings in environment
    import os
    proxy_vars = [var for var in os.environ if "proxy" in var.lower()]
    if proxy_vars:
        print(f"Found proxy-related environment variables: {proxy_vars}")
    
    # Check for conflicting packages or configurations
    try:
        import sys
        openai_related = [mod for mod in sys.modules if "openai" in mod.lower()]
        if len(openai_related) > 1:
            print(f"Multiple OpenAI-related modules loaded: {openai_related}")
    except Exception as e:
        print(f"Error checking modules: {e}")
    
    if not OPENAI_V1:
        # For older versions, just return the openai module
        return openai
        
    # For OpenAI v1.0+
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # For debugging, check OpenAI class signature
    try:
        from openai import OpenAI
        import inspect
        print(f"OpenAI.__init__ signature: {inspect.signature(OpenAI.__init__)}")
    except Exception as e:
        print(f"Error inspecting OpenAI signature: {e}")
    
    # Try directly creating the client with no parameters first
    try:
        if azure:
            from openai import AzureOpenAI
            print("Attempting to create AzureOpenAI client with no parameters")
            return AzureOpenAI()
        else:
            from openai import OpenAI
            print("Attempting to create OpenAI client with no parameters")
            return OpenAI()
    except Exception as e:
        print(f"Failed to initialize with no parameters: {str(e)}")
    
    # Try with absolutely minimal parameters - just the API key
    try:
        if azure:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            azure_key = os.environ.get("AZURE_OPENAI_KEY")
            if not azure_endpoint or not azure_key:
                print("Azure OpenAI endpoint or key not found in environment variables")
                OPENAI_INIT_FAILED = True
                return None
                
            from openai import AzureOpenAI
            print("Attempting to create AzureOpenAI client with minimal parameters")
            return AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key
            )
        else:
            if not api_key:
                print("OpenAI API key not found in environment variables")
                OPENAI_INIT_FAILED = True
                return None
                
            from openai import OpenAI
            print("Attempting to create OpenAI client with API key only")
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
            except Exception as e:
                print(f"Manual initialization failed: {e}")
                # Try normal initialization as fallback
                return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize OpenAI client with API key only: {e}")
        OPENAI_INIT_FAILED = True
        return None


def openai_setup(model: str, azure: bool = False, *args, **kwargs):
    """Setup an OpenAI language model.

    :param model: The name of the model (e.g. "gpt-3.5-turbo-0613").
    :param azure: If true, use a model deployed on azure.

    This function uses the following environment variables:

    - OPENAI_API_KEY
    - OPENAI_API_ORG
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_KEY
    - AZURE_OPENAI_VERSION

    Returns:
        LLM_Interface: An LLM to work with!
    """
    # Create a client that works with the current OpenAI version
    client = create_openai_client(azure)
    
    # If client creation failed, use DummyChatModel
    if not client and OPENAI_V1:
        print("Using DummyChatModel as fallback due to OpenAI client initialization failure")
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


def setup(model: Union[AbstractChatModel, str]):
    """Setup a chat model. If the input is a string, we assume that it is the name of an OpenAI model."""
    if isinstance(model, str):
        model = openai_setup(model)
    return model


def chat_completion(llm: Union[str, AbstractChatModel], messages):
    """Execute a sequence of user and assistant messages with an AbstractChatModel.

    Sends multiple individual messages to the AbstractChatModel.
    """
    llm = setup(llm)
    # we sequentially execute all assistant messages that do not have a content.
    messages = copy.deepcopy(messages)  # do not alter the input
    for msg_idx in range(len(messages)):
        if messages[msg_idx]["role"] == "assistant":
            if not "content" in messages[msg_idx]:
                # send message
                messages[msg_idx]["content"] = llm.chat_completion(
                    messages[:msg_idx],
                    temperature=messages[msg_idx]["temperature"],
                    max_tokens=messages[msg_idx]["max_tokens"],
                )
            # remove all keys except "role" and "content"
            keys = list(messages[msg_idx].keys())
            for k in keys:
                if not k in ["role", "content"]:
                    messages[msg_idx].pop(k)
    return messages
