"""
Misc. helper functions
"""

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import openai

# Try to import from llm module first to keep client initialization consistent
try:
    from .llm import create_openai_client, OPENAI_V1, OPENAI_INIT_FAILED, DummyChatModel
    client = create_openai_client()
except ImportError:
    # Fallback to direct initialization
    try:
        # For OpenAI v1.0+
        from openai import OpenAI
        # Initialize client with only supported parameters
        try:
            client = OpenAI()
        except TypeError as e:
            print(f"OpenAI client initialization error: {e}")
            try:
                # Try with just the API key
                import os
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    client = OpenAI(api_key=api_key)
                else:
                    client = None
            except Exception:
                # If that fails too, use None
                client = None
        OPENAI_V1 = True
    except ImportError:
        # For OpenAI <v1.0
        client = openai
        OPENAI_V1 = False
        # Print OpenAI version for debugging
        print(f"Using OpenAI API version: {openai.__version__}")

import tiktoken


def num_tokens_from_string_(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# @retry(
#    retry=retry_if_not_exception_type(openai.InvalidRequestError),
#    wait=wait_random_exponential(min=1, max=60),
#    stop=stop_after_attempt(10),
# )
def openai_completion_query(model, messages, **kwargs):
    """Catches exceptions and retries, good for deployment / running experiments"""
    # Check if OpenAI client failed to initialize
    if client is None:
        return "Error: OpenAI client failed to initialize. Please check your API key."
        
    if OPENAI_V1:
        try:
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return f"Error: {str(e)}"
    else:
        try:
            # Legacy OpenAI API for versions around 0.27.x
            if hasattr(client, "ChatCompletion"):
                response = client.ChatCompletion.create(model=model, messages=messages, **kwargs)
                # Extract content based on structure
                try:
                    if hasattr(response.choices[0], "message"):
                        return response.choices[0].message["content"]
                    else:
                        return response.choices[0]["message"]["content"]
                except Exception as e:
                    print(f"Error extracting content: {e}")
                    print(f"Response structure: {response}")
                    return ""
            else:
                # For very old versions
                print("Using direct completion API for older OpenAI versions")
                prompt = _format_messages_as_prompt(messages)
                response = client.Completion.create(engine=model, prompt=prompt, **kwargs)
                return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error with OpenAI API call: {str(e)}")
            try:
                # Try with api_resources as a fallback
                from openai.api_resources import ChatCompletion
                response = ChatCompletion.create(
                    api_key=openai.api_key,
                    model=model,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message["content"]
            except Exception as e2:
                print(f"Fallback also failed: {str(e2)}")
                return f"Error: Could not generate response with OpenAI API version {openai.__version__}. Please update to OpenAI SDK v1.0.0 or higher."


def openai_debug_completion_query(model, messages, **kwargs):
    """Does not catch exceptions, better for debugging"""
    # Check if OpenAI client failed to initialize
    if client is None:
        return "Error: OpenAI client failed to initialize. Please check your API key."
    
    if OPENAI_V1:
        try:
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return f"Error: {str(e)}"
    else:
        try:
            # Legacy OpenAI API for versions around 0.27.x
            if hasattr(client, "ChatCompletion"):
                response = client.ChatCompletion.create(model=model, messages=messages, **kwargs)
                # Extract content based on structure
                if hasattr(response.choices[0], "message"):
                    return response.choices[0].message["content"]
                else:
                    return response.choices[0]["message"]["content"]
            else:
                # For very old versions
                print("Using direct completion API for older OpenAI versions")
                prompt = _format_messages_as_prompt(messages)
                response = client.Completion.create(engine=model, prompt=prompt, **kwargs)
                return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error with OpenAI API call: {str(e)}")
            # Try with api_resources as a fallback
            from openai.api_resources import ChatCompletion
            response = ChatCompletion.create(
                api_key=openai.api_key,
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message["content"]


def _format_messages_as_prompt(messages):
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


def parse_guidance_query(query):
    """This is a utility function that parses a guidance query string into a list of messages for the openai api.

    It only works with the most simple types of queries.
    """
    messages = []
    start_tokens = ["{{#system~}}", "{{#assistant~}}", "{{#user~}}"]
    # find first occurence of any start toke in the query
    position = -1
    next_token = None
    for token in start_tokens:
        next_position = query.find(token)
        if next_position != -1 and (position == -1 or next_position < position):
            position = next_position
            next_token = token
    if next_token == start_tokens[0]:  # system
        end_pos = query.find("{{~/system}}")
        messages.append(
            {
                "role": "system",
                "content": query[position + len(start_tokens[0]) : end_pos].strip(),
            }
        )
    if next_token == start_tokens[1]:  # assistant
        end_pos = query.find("{{~/assistant}}")
        messages.append(
            {
                "role": "assistant",
                "content": query[position + len(start_tokens[1]) : end_pos].strip(),
            }
        )
    if next_token == start_tokens[2]:  # user
        end_pos = query.find("{{~/user}}")
        messages.append(
            {
                "role": "user",
                "content": query[position + len(start_tokens[2]) : end_pos].strip(),
            }
        )
    if next_token is not None and len(query[end_pos:]) > 15:
        messages.extend(parse_guidance_query(query[end_pos:]))
    return messages


# Create a function to help users directly create a working client
def create_direct_client(api_key=None):
    """
    Utility function to create an OpenAI client directly.
    
    This bypasses all package initializations and creates a direct client
    for users who are experiencing issues with the automatic initialization.
    
    Args:
        api_key (str, optional): Your OpenAI API key. If not provided, will try to get from environment.
    
    Returns:
        A working OpenAI client or None if initialization fails.
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
            import os
            api_key = os.environ.get("OPENAI_API_KEY")
            
        if not api_key:
            print("No API key provided or found in environment variables.")
            return None
            
        # Create client based on available API
        if v1_api_available:
            print("Creating OpenAI v1.x client...")
            try:
                # Try the most direct way possible
                client = OpenAI(api_key=api_key)
                return client
            except Exception as e:
                print(f"Standard initialization failed: {e}")
                # Try bare-bones approach using manual initialization
                print("Trying manual client creation...")
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
                    print("Manual client creation successful")
                    return client
                except Exception as e:
                    print(f"Manual creation failed: {e}")
                    return None
        else:
            print("Creating OpenAI v0.x client...")
            # For older versions
            openai.api_key = api_key
            return openai
    except Exception as e:
        print(f"Failed to create direct client: {e}")
        return None
