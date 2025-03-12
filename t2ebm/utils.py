"""
Misc. helper functions
"""

import openai
import os
import tiktoken

# Import necessary modules and classes from llm.py
try:
    from .llm import (create_openai_client, OPENAI_V1, OPENAI_INIT_FAILED, 
                     OpenAIInitializationError, OpenAICompletionError,
                     LocalLLMError)
    # Don't initialize the client at import time - defer until needed
    client = None
    client_initialized = False
except ImportError:
    # Fallback to direct initialization
    try:
        # For OpenAI v1.0+
        from openai import OpenAI
        # Don't initialize client at import time
        client = None
        client_initialized = False
        OPENAI_V1 = True
    except ImportError:
        # For OpenAI <v1.0
        client = None  # Don't initialize
        client_initialized = False
        import openai
        OPENAI_V1 = False
    
    # Define exception classes if importing from llm failed
    class OpenAIInitializationError(Exception):
        """Exception raised when OpenAI client initialization fails."""
        pass
    
    class OpenAICompletionError(Exception):
        """Exception raised when OpenAI completion fails."""
        pass
    
    class LocalLLMError(Exception):
        """Exception raised when a local LLM request fails."""
        pass


def _ensure_client():
    """
    Lazily initialize the OpenAI client only when needed.
    This function is called before any API operations.
    
    Returns:
        The initialized client
        
    Raises:
        OpenAIInitializationError: If initialization fails
    """
    global client, client_initialized
    
    # If already initialized, just return the client
    if client_initialized:
        return client
    
    # Try to initialize the client
    try:
        from .llm import create_openai_client
        client = create_openai_client()
        client_initialized = True
        return client
    except (ImportError, OpenAIInitializationError):
        # Fallback to direct initialization
        try:
            # For OpenAI v1.0+
            from openai import OpenAI
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                client = OpenAI(api_key=api_key)
                client_initialized = True
                return client
            
            # Try without API key (in case it's in config file)
            try:
                client = OpenAI()
                client_initialized = True
                return client
            except:
                pass
                
        except ImportError:
            # For OpenAI <v1.0
            if "OPENAI_API_KEY" in os.environ:
                openai.api_key = os.environ["OPENAI_API_KEY"]
                client = openai
                client_initialized = True
                return client
                
    # If we got here, we couldn't initialize the client
    client_initialized = True  # Mark as initialized to avoid repeated attempts
    return None


def num_tokens_from_string_(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def openai_completion_query(model, messages, **kwargs):
    """
    Makes a completion query to the OpenAI API with error handling.
    
    Args:
        model: The model to use for completion
        messages: The messages to send to the model
        **kwargs: Additional arguments to pass to the API
        
    Returns:
        str: The model's response text
        
    Raises:
        OpenAIInitializationError: If the client is not initialized
        OpenAICompletionError: If the completion request fails
    """
    # Lazily initialize the client
    global client
    if not client_initialized:
        client = _ensure_client()
    
    # Check if OpenAI client failed to initialize
    if client is None:
        raise OpenAIInitializationError("OpenAI client failed to initialize. Please check your API key.")
    
    # Handle OpenAI v1.0+ API
    if OPENAI_V1:
        try:
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise OpenAICompletionError(f"Error with OpenAI API call: {str(e)}")
    
    # Handle older OpenAI API versions
    try:
        # Try ChatCompletion API if available
        if hasattr(client, "ChatCompletion"):
            response = client.ChatCompletion.create(model=model, messages=messages, **kwargs)
            try:
                if hasattr(response.choices[0], "message"):
                    return response.choices[0].message["content"]
                return response.choices[0]["message"]["content"]
            except Exception as e:
                raise OpenAICompletionError(f"Error extracting content from response: {str(e)}")
        
        # Fall back to Completion API for very old versions
        prompt = _format_messages_as_prompt(messages)
        response = client.Completion.create(engine=model, prompt=prompt, **kwargs)
        return response.choices[0].text.strip()
    except Exception as e:
        # Last resort: try through api_resources if available
        return _try_api_resources_fallback(model, messages, **kwargs)


def openai_debug_completion_query(model, messages, **kwargs):
    """
    Makes a completion query to the OpenAI API with minimal error handling.
    Better for debugging issues.
    
    Args:
        model: The model to use for completion
        messages: The messages to send to the model
        **kwargs: Additional arguments to pass to the API
        
    Returns:
        str: The model's response text
        
    Raises:
        OpenAIInitializationError: If the client is not initialized
        OpenAICompletionError: If the completion request fails
    """
    # Lazily initialize the client
    global client
    if not client_initialized:
        client = _ensure_client()
    
    # Check if OpenAI client failed to initialize
    if client is None:
        raise OpenAIInitializationError("OpenAI client failed to initialize. Please check your API key.")
    
    # Handle OpenAI v1.0+ API
    if OPENAI_V1:
        try:
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise OpenAICompletionError(f"Error with OpenAI API call: {str(e)}")
    
    # Handle older OpenAI API versions
    try:
        # Try ChatCompletion API if available
        if hasattr(client, "ChatCompletion"):
            response = client.ChatCompletion.create(model=model, messages=messages, **kwargs)
            if hasattr(response.choices[0], "message"):
                return response.choices[0].message["content"]
            return response.choices[0]["message"]["content"]
        
        # Fall back to Completion API for very old versions
        prompt = _format_messages_as_prompt(messages)
        response = client.Completion.create(engine=model, prompt=prompt, **kwargs)
        return response.choices[0].text.strip()
    except Exception as e:
        # Last resort: try through api_resources if available
        return _try_api_resources_fallback(model, messages, **kwargs)


def _try_api_resources_fallback(model, messages, **kwargs):
    """
    Helper function to try the api_resources fallback approach.
    Used by both completion query functions.
    
    Args:
        model: The model to use
        messages: The messages to send
        **kwargs: Additional arguments
        
    Returns:
        str: Response if successful
        
    Raises:
        OpenAICompletionError: If fallback fails
    """
    # Check if api_resources exists
    api_resources_exists = False
    try:
        import importlib
        api_resources = importlib.import_module("openai.api_resources")
        api_resources_exists = hasattr(api_resources, "ChatCompletion")
    except (ImportError, AttributeError):
        api_resources_exists = False
    
    if api_resources_exists:
        # Only import if it exists
        try:
            from openai.api_resources import ChatCompletion
            response = ChatCompletion.create(
                api_key=openai.api_key,
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message["content"]
        except Exception as e:
            raise OpenAICompletionError(f"Error with api_resources fallback: {str(e)}")
            
    raise OpenAICompletionError(
        f"Could not complete request. This version of OpenAI SDK ({openai.__version__}) "
        f"doesn't support the required API methods."
    )


def _format_messages_as_prompt(messages):
    """
    Convert chat messages to a single prompt string for older API versions.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
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


def parse_guidance_query(query):
    """
    Parse a guidance query string into a list of messages for the OpenAI API.
    
    This utility function only works with simple types of queries.
    
    Args:
        query: Guidance query string
        
    Returns:
        list: Messages in OpenAI API format
    """
    messages = []
    start_tokens = ["{{#system~}}", "{{#assistant~}}", "{{#user~}}"]
    
    # Find first occurrence of any start token in the query
    position = -1
    next_token = None
    for token in start_tokens:
        next_position = query.find(token)
        if next_position != -1 and (position == -1 or next_position < position):
            position = next_position
            next_token = token
    
    # Extract message based on token type
    if next_token == start_tokens[0]:  # system
        end_pos = query.find("{{~/system}}")
        messages.append({
            "role": "system",
            "content": query[position + len(start_tokens[0]) : end_pos].strip(),
        })
    elif next_token == start_tokens[1]:  # assistant
        end_pos = query.find("{{~/assistant}}")
        messages.append({
            "role": "assistant",
            "content": query[position + len(start_tokens[1]) : end_pos].strip(),
        })
    elif next_token == start_tokens[2]:  # user
        end_pos = query.find("{{~/user}}")
        messages.append({
            "role": "user",
            "content": query[position + len(start_tokens[2]) : end_pos].strip(),
        })
    
    # Process remaining part of the query recursively
    if next_token is not None and len(query[end_pos:]) > 15:
        messages.extend(parse_guidance_query(query[end_pos:]))
    
    return messages


def create_direct_client(api_key=None):
    """
    Utility function to create an OpenAI client directly.
    
    This bypasses all package initializations and creates a direct client
    for users who are experiencing issues with the automatic initialization.
    
    Args:
        api_key (str, optional): Your OpenAI API key. If not provided, will try to get from environment.
    
    Returns:
        A working OpenAI client
        
    Raises:
        OpenAIInitializationError: If client creation fails
    """
    try:
        # Try to determine if we can use OpenAI v1 API
        try:
            from openai import OpenAI
            v1_api_available = True
        except ImportError:
            v1_api_available = False
            
        # Get API key if not provided
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        if not api_key:
            raise OpenAIInitializationError("API key is required. Provide as parameter or set OPENAI_API_KEY environment variable.")
            
        # Create client based on available API
        if v1_api_available:
            try:
                # Try the most direct way possible
                client = OpenAI(api_key=api_key)
                return client
            except Exception as e:
                # Try bare-bones approach using manual initialization
                try:
                    # Create object directly to bypass __init__
                    client = object.__new__(OpenAI)
                    
                    # Manually set required attributes
                    import httpx
                    http_client = httpx.Client(
                        base_url="https://api.openai.com/v1",
                        headers={"Authorization": f"Bearer {api_key}"}
                    )
                    setattr(client, "_client", http_client)
                    setattr(client, "api_key", api_key)
                    return client
                except Exception as e2:
                    raise OpenAIInitializationError(f"Failed to create OpenAI client: {str(e2)}")
        else:
            # For older versions
            openai.api_key = api_key
            return openai
    except Exception as e:
        raise OpenAIInitializationError(f"Failed to create OpenAI client: {str(e)}")
