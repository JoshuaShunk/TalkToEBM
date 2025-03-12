"""
Prompts that ask the LLM to perform tasks with Graphs and EBMs.

Functions either return a string or a sequene of messages / desired responses in the OpenAI message format.
"""


def graph_system_msg(expert_description="an expert statistician and data scientist"):
    """A system message that instructs the LLM to work with the graphs of an EBM.

    Args:
        expert_description (str, optional): Description of the expert that we want the LLM to be. Defaults to "an expert statistician and data scientist".

    Returns:
        str: The system message.
    """
    return f"""You are {expert_description}. Your task is to analyze and explain the text-based representation of graphs produced by a Generalized Additive Model (GAM).

IMPORTANT: The graph data is provided to you as text (JSON format) in the user's message. You must analyze this text-based data directly - there is NO visual graph image to look at. The text representation contains all the information needed for your analysis.

Provide direct, factual analysis without:
1. Adding conversational phrases like 'I'd be happy to provide' or 'Please let me know if you need more information'
2. Asking to see a visual representation of the graph
3. Saying you cannot analyze without seeing the actual graph

Focus only on the technical content and analysis of the data provided in text format."""


def describe_graph(
    graph: str,
    graph_description="",
    dataset_description="",
    task_description="Analyze the data provided in this text-based graph representation and describe the general pattern.",
):
    """Prompt the LLM to describe a graph. This is intended to be the first prompt in a conversation about a graph.

    Args:
        graph (str): The graph to describe (in JSON format, obtained from graph_to_text).
        graph_description (str, optional): Additional description of the graph (e.g. "The y-axis of the graph depicts the probability of sucess."). Defaults to "".
        dataset_description (str, optional): Additional description of the dataset (e.g. "The dataset is a Pneumonia dataset collected by [...]"). Defaults to "".
        task_description (str, optional): A final prompt to instruct the LLM. Defaults to "Analyze the data provided in this text-based graph representation and describe the general pattern.".

    Returns:
        str: The prompt to describe the graph.
    """   
    prompt = """Below is the text-based representation of a Generalized Additive Model (GAM) graph. The graph data is presented as a JSON object with keys representing the x-axis and values representing the y-axis. For continuous features, the keys are intervals that represent ranges where the function predicts the same value. For categorical features, each key represents a possible value that the feature can take.

IMPORTANT: This is the complete graph information that you need to analyze. There is no additional visual representation - all analysis should be based on this text data.
    
The graph data is provided in the following format:
    - The name of the feature depicted in the graph
    - The type of the feature (continuous, categorical, or boolean)
    - Mean values (these are the y-axis values for each x value or range)
    - Lower bounds of confidence interval (optional)
    - Upper bounds of confidence interval (optional)\n\n"""

    # the graph
    prompt += f"Here is the graph data:\n\n{graph}\n\n"

    # optional graph_description
    if graph_description is not None and len(graph_description) > 0:
        prompt += f"{graph_description}\n\n"

    # optional dataset description
    if dataset_description is not None and len(dataset_description) > 0:
        prompt += f"Here is a description of the dataset that the model was trained on:\n\n{dataset_description}\n\n"

    # the task that the LLM is intended to perform
    prompt += task_description
    return prompt


def describe_graph_cot(graph, num_sentences=7, **kwargs):
    """Use chain-of-thought reasoning to elicit a description of a graph in at most {num_sentences} sentences.

    Returns:
        Messages in OpenAI format.
    """
    return [
        {"role": "system", "content": graph_system_msg()},
        {"role": "user", "content": describe_graph(graph, **kwargs)},
        {"role": "assistant", "temperature": 0.7, "max_tokens": 3000},
        {
            "role": "user",
            "content": "Based on the text-based graph data I provided (not a visual graph), analyze the data carefully and identify any regions you find surprising or counterintuitive. Analyze potential explanations for these patterns.",
        },
        {"role": "assistant", "temperature": 0.7, "max_tokens": 2000},
        {
            "role": "user",
            "content": f"Based on the text-based graph data I provided earlier, give a direct, factual description in at most {num_sentences} sentences. Include important surprising patterns. Focus only on the technical content without asking for visual information or adding conversational phrases.",
        },
        {"role": "assistant", "temperature": 0.7, "max_tokens": 2000},
    ]


def summarize_ebm(
    feature_importances: str,
    graph_descriptions: str,
    expert_description="an expert statistician and data scientist",
    dataset_description="",
    num_sentences: int = None,
):
    """Prompt the LLM to summarize a Generalized Additive Model (GAM).

    Returns:
        Messages in OpenAI format.
    """
    messages = [
        {
            "role": "system",
            "content": f"""You are {expert_description}. Your task is to provide an overall summary of a Generalized Additive Model (GAM) based on text data. The model consists of different graphs that contain the effect of a specific input feature.

IMPORTANT: All data is provided as text - there are no visual graphs to look at. You have all the information needed to provide your analysis.

Provide direct, factual analysis without adding conversational phrases or asking for visual representations.""",
        }
    ]
    user_msg = """Your task is to summarize a Generalized Additive Model (GAM) based on the text data provided. To perform this task, you are given:
    - The global feature importances of the different features in the model.
    - Textual summaries of the graphs for the different features in the model. There is exactly one graph for each feature in the model. """
    user_msg += f"Here are the global feature importances.\n\n{feature_importances}\n\n"
    user_msg += f"Here are the descriptions of the different graphs.\n\n{graph_descriptions}\n\n"
    if dataset_description is not None and len(dataset_description) > 0:
        user_msg += f"Here is a description of the dataset that the model was trained on.\n\n{dataset_description}\n\n"
    user_msg += """Now, provide a direct summary of the model based on this text data.
    
The summary should contain the most important features in the model and their effect on the outcome. Unimportant effects and features can be ignored. 
    
Pay special attention to include any surprising patterns in the summary. Focus only on the technical content without adding any conversational phrases or requesting visual graphs."""
    messages.append({"role": "user", "content": user_msg})
    messages.append({"role": "assistant", "temperature": 0.7, "max_tokens": 3000})
    if num_sentences is not None:
        messages.append(
            {
                "role": "user",
                "content": f"Shorten the above summary to at most {num_sentences} sentences. Keep only the most important information and avoid any conversational phrases or requests for visual data.",
            }
        )
        messages.append({"role": "assistant", "temperature": 0.7, "max_tokens": 2000})
    return messages
