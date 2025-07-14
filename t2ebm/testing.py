"""
Testing utilities for TalkToEBM.

This module contains classes and functions that are only intended for testing
and development purposes. These should not be used in production code.
"""

from typing import Any, Dict, List

from t2ebm.llm import AbstractChatModel
from t2ebm.utils import OpenAIInitializationError, _ensure_client


class DummyChatModel(AbstractChatModel):
    """
    Dummy chat model for testing purposes.

    Always returns a simple test response. This should only be used
    in tests and development - never in production code.

    Example:
        >>> from t2ebm.testing import DummyChatModel
        >>> model = DummyChatModel()
        >>> response = model.chat_completion([], 0.5, 100)
        >>> print(response)
        This is a dummy response for testing purposes.
    """

    def chat_completion(
        self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> str:
        """
        Return a dummy response for testing.

        Args:
            messages: The messages (ignored)
            temperature: Sampling temperature (ignored)
            max_tokens: Maximum tokens (ignored)

        Returns:
            str: A dummy response
        """
        return "This is a dummy response for testing purposes."


def openai_debug_completion_query(model, messages, **kwargs):
    """
    Makes a completion query to the OpenAI API with minimal error handling.

    This is a debug utility for testing purposes only. It provides less
    robust error handling than the production functions and should not
    be used in production code.

    Args:
        model: The model to use for completion
        messages: The messages to send to the model
        **kwargs: Additional arguments to pass to the API

    Returns:
        str: The completion response

    Raises:
        OpenAIInitializationError: If the client is not initialized
    """
    try:
        client = _ensure_client()
    except Exception:
        raise OpenAIInitializationError("Failed to initialize OpenAI client")

    if client is None:
        raise OpenAIInitializationError("OpenAI client is None")

    # Import the OPENAI_V1 flag to check API version
    from t2ebm.utils import OPENAI_V1

    # Use the minimal error handling approach for debugging
    if OPENAI_V1 and hasattr(client, "chat") and hasattr(client.chat, "completions"):
        # OpenAI v1.0+ API
        response = client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content
    elif hasattr(client, "ChatCompletion"):
        # Legacy OpenAI API with ChatCompletion
        response = client.ChatCompletion.create(
            model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content
    else:
        # Fallback to Completion API (legacy)
        from t2ebm.utils import _format_messages_as_prompt

        prompt = _format_messages_as_prompt(messages)
        response = client.Completion.create(model=model, prompt=prompt, **kwargs)
        return response.choices[0].text.strip()
