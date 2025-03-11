"""
Example workaround for OpenAI client initialization issues in Colab.

This script demonstrates how to manually create a working LLM client
if you're experiencing the 'proxies' error in Colab.
"""

import os
import sys
import openai

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"  # Replace with your actual key

# Import t2ebm modules
import t2ebm
from t2ebm.llm import OpenAIChatModel, DummyChatModel

# Option 1: Create a direct client using the utility function
try:
    # Check if the utility function exists
    if hasattr(t2ebm.utils, "create_direct_client"):
        direct_client = t2ebm.utils.create_direct_client()
        if direct_client:
            print("Successfully created a direct OpenAI client!")
            # Create a custom LLM interface with this client
            custom_llm = OpenAIChatModel(direct_client, "gpt-4-turbo-2024-04-09")
            
            # Now we can use this with any t2ebm function
            # Example (uncomment to use):
            # graph_description = t2ebm.describe_graph(custom_llm, ebm, 0)
            # print(graph_description)
except Exception as e:
    print(f"Option 1 failed: {e}")

# Option 2: Use a custom minimal client
try:
    # Try to create a minimal client directly
    from openai import OpenAI
    
    # The simplest initialization possible
    minimal_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    custom_llm = OpenAIChatModel(minimal_client, "gpt-4-turbo-2024-04-09")
    
    # Example (uncomment to use):
    # graph_description = t2ebm.describe_graph(custom_llm, ebm, 0)
    # print(graph_description)
except Exception as e:
    print(f"Option 2 failed: {e}")

# Option 3: Use the dummy model for testing/debugging
try:
    dummy_llm = DummyChatModel()
    
    # Example (uncomment to use):
    # graph_description = t2ebm.describe_graph(dummy_llm, ebm, 0)
    # print(graph_description)
except Exception as e:
    print(f"Option 3 failed: {e}")

print("\nInstructions for fixing OpenAI client issues:")
print("1. Make sure you're using a version of OpenAI that matches your code")
print("2. If using OpenAI v1+, make sure it's completely updated: pip install --upgrade openai")
print("3. If using an older OpenAI version, try using that version consistently")
print("4. Check for proxy settings or environment variables that might be interfering")
print("5. Try passing your own client directly to functions as shown in this example") 