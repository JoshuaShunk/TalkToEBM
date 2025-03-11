"""
TalkToEBM structures conversations in a generic OpenAI message format that can be executed with different LLMs.

We interface the LLM via the simple class AbstractChatModel. To use your own LLM, simply implement the chat_completion method in a subclass.
"""

from dataclasses import dataclass
import copy
import os
import sys
from typing import Union

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
        if OPENAI_V1:
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
                # Fallback to a more direct approach
                try:
                    from openai.api_resources import ChatCompletion
                    response = ChatCompletion.create(
                        api_key=openai.api_key,
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    response_content = response.choices[0].message["content"]
                except Exception as e2:
                    print(f"Fallback also failed: {str(e2)}")
                    response_content = f"Error: Could not generate response with OpenAI API version {openai.__version__}. Please update to OpenAI SDK v1.0.0 or higher."
        
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
    if OPENAI_V1:
        # Filter kwargs to only include parameters that are actually supported
        # Remove common parameters that are known to cause issues
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key not in ['proxies']:  # Add other problematic params here
                filtered_kwargs[key] = value
            else:
                print(f"Removing unsupported parameter '{key}' for OpenAI client")

        try:
            if azure:  # azure deployment
                azure_params = {
                    "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    "api_key": os.environ.get("AZURE_OPENAI_KEY"),
                    "api_version": os.environ.get("AZURE_OPENAI_VERSION")
                }
                # Filter out None values
                azure_params = {k: v for k, v in azure_params.items() if v is not None}
                client = AzureOpenAI(**azure_params, **filtered_kwargs)
            else:  # openai api
                openai_params = {
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "organization": os.environ.get("OPENAI_API_ORG")
                }
                # Filter out None values
                openai_params = {k: v for k, v in openai_params.items() if v is not None}
                client = OpenAI(**openai_params, **filtered_kwargs)
        except TypeError as e:
            print(f"OpenAI client initialization error: {e}")
            print("Attempting to initialize with minimal parameters...")
            try:
                if azure:
                    client = AzureOpenAI(
                        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                        api_key=os.environ.get("AZURE_OPENAI_KEY")
                    )
                else:
                    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except Exception as e2:
                print(f"Minimal initialization also failed: {e2}")
                print("Initializing with no parameters as last resort")
                try:
                    if azure:
                        client = AzureOpenAI()
                    else:
                        client = OpenAI()
                except Exception as e3:
                    print(f"Failed to initialize OpenAI client: {e3}")
                    print("Using DummyChatModel as fallback")
                    return DummyChatModel()
    else:
        # Legacy OpenAI API
        client = openai
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

    # the llm
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
