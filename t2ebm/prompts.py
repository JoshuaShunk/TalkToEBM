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
    return f"""You are {expert_description}. You interpret global explanations produced by a Generalized Additive Model (GAM). 

IMPORTANT: When analyzing graphs, you must ONLY describe patterns that are explicitly present in the provided data. Never fabricate or invent data points, trends, time periods, categories, or relationships that are not clearly represented in the graph. 

If the graph shows numeric ranges rather than specific categories, do not assume these represent time periods, years, or any other specific units unless explicitly stated in the description.

Always ground your analysis in the exact data provided, and be transparent about uncertainty. If you're unsure about what a specific range or value represents, acknowledge this uncertainty rather than making assumptions.

You MUST ALWAYS explicitly mention the feature name in your description and relate the patterns to this specific feature. For example, if describing an 'Age' feature, begin your description with "The Age feature shows..." or similar phrasing that clearly identifies the feature being described.

You answer all questions to the best of your ability, relying ONLY on the graphs provided by the user, any other information you are given, and your knowledge about the real world."""


def describe_graph(
    graph: str,
    graph_description="",
    dataset_description="",
    task_description="Please describe the general pattern of the graph.",
):
    """Prompt the LLM to describe a graph. This is intended to be the first prompt in a conversation about a graph.

    Args:
        graph (str): The graph to describe (in JSON format, obtained from graph_to_text).
        graph_description (str, optional): Additional description of the graph (e.g. "The y-axis of the graph depicts the probability of sucess."). Defaults to "".
        dataset_description (str, optional): Additional description of the dataset (e.g. "The dataset is a Pneumonia dataset collected by [...]"). Defaults to "".
        task_description (str, optional): A final prompt to instruct the LLM. Defaults to "Please describe the general pattern of the graph.".

    Returns:
        str: The prompt to describe the graph.
    """   
    prompt = """Below is the graph of a Generalized Additive Model (GAM). The graph is presented as a JSON object with keys representing the x-axis and values representing the y-axis. For continuous features, the keys are intervals that represent ranges where the function predicts the same value. For categorical features, each key represents a possible value that the feature can take.
    
The graph is provided in the following format:
    - The name of the feature depicted in the graph
    - The type of the feature (continuous, categorical, or boolean)
    - Mean values
    - Lower bounds of confidence interval (optional)
    - Upper bounds of confidence interval (optional)

CRITICAL INSTRUCTIONS:
1. ALWAYS begin your description by explicitly mentioning the feature name
2. Consistently refer to the feature by name throughout your description
3. Only describe patterns that are explicitly shown in the provided data
4. Do NOT fabricate or invent any data points, trends, or relationships
5. For continuous features, do not assume that numeric ranges represent specific time periods, years, or dates unless explicitly stated
6. When uncertain about what a value represents, acknowledge this uncertainty rather than making assumptions
7. Ground all observations in the exact values provided in the graph data
8. If you need context that is not provided, state this clearly rather than making up information

"""

    # the graph
    prompt += f"Here is the graph:\n\n{graph}\n\n"

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
        {"role": "assistant", "temperature": 0.3, "max_tokens": 3000},
        {
            "role": "user",
            "content": """First, please itemize and list ONLY the specific data points and ranges that you can directly observe in the graph. 
Include:
1. The name and type of the feature
2. The exact ranges or categories shown on the x-axis
3. The corresponding values on the y-axis
4. The confidence intervals if provided

DO NOT interpret the data yet, just summarize the actual values you observe."""
        },
        {"role": "assistant", "temperature": 0.3, "max_tokens": 2000},
        {
            "role": "user",
            "content": "Great, now please study the graph carefully and highlight any regions you may find surprising or counterintuitive. You may also suggest an explanation for why this behavior is surprising, and what may have caused it. Remember to refer only to the exact data you observed in your previous response, and do not invent or assume data not present in the graph.",
        },
        {"role": "assistant", "temperature": 0.3, "max_tokens": 2000},
        {
            "role": "user",
            "content": f"""Thanks. Now please provide a brief, at most {num_sentences} sentence description of the graph. Be sure to include any important surprising patterns in the description. 

IMPORTANT REQUIREMENTS:
- BEGIN your description by explicitly mentioning the feature name (e.g., "The Age feature shows...")
- CONTINUE to reference the feature by name throughout your description
- Refer ONLY to the data points and patterns you explicitly listed in your first response
- DO NOT fabricate or invent trends, time periods, or relationships not shown in the data
- If the meaning of a range is unclear, acknowledge this rather than making assumptions
- Use direct references to specific values when making claims about the data

You can assume that the user knows that the graph is from a Generalized Additive Model (GAM).""",
        },
        {"role": "assistant", "temperature": 0.3, "max_tokens": 2000},
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
            "content": f"""You are {expert_description}. Your task is to provide an overall summary of a Generalized Additive Model (GAM). The model consists of different graphs that contain the effect of a specific input feature.

IMPORTANT: When writing your summary, you must ONLY describe patterns and relationships that are explicitly present in the provided data. Never fabricate or invent data points, trends, time periods, categories, or relationships that are not clearly represented in the information provided.

Always ground your analysis in the exact data provided, and be transparent about uncertainty. If you're unsure about what a specific range or value represents, acknowledge this uncertainty rather than making assumptions.

When discussing each feature, ALWAYS explicitly mention the feature by name (e.g., "The Age feature shows..." or "Blood Pressure is associated with...") rather than using generic descriptions.""",
        }
    ]
    user_msg = """Your task is to summarize a Generalized Additive Model (GAM). To perform this task, you will be given
    - The global feature importances of the different features in the model.
    - Summaries of the graphs for the different features in the model. There is exactly one graph for each feature in the model. 

INSTRUCTIONS:
1. ALWAYS refer to each feature by its specific name when describing its effects
2. Only reference information explicitly present in the provided data
3. Do NOT fabricate or invent any data points, trends, or relationships
4. For continuous features, do not assume numeric ranges represent specific time periods, years, or dates
5. When uncertain about what a value represents, acknowledge this uncertainty rather than making assumptions
6. Ground all observations in the exact values provided in the feature data
"""
    user_msg += f"Here are the global feature importances.\n\n{feature_importances}\n\n"
    user_msg += f"Here are the descriptions of the different graphs.\n\n{graph_descriptions}\n\n"
    if dataset_description is not None and len(dataset_description) > 0:
        user_msg += f"Here is a description of the dataset that the model was trained on.\n\n{dataset_description}\n\n"
    user_msg += """Now, please provide a summary of the model.
    
The summary should contain the most important features in the model and their effect on the outcome. Unimportant effects and features can be ignored. 
    
Pay special attention to include any surprising patterns in the summary. Make sure to identify each feature by its specific name when describing its effects."""
    messages.append({"role": "user", "content": user_msg})
    messages.append({"role": "assistant", "temperature": 0.3, "max_tokens": 3000})
    if num_sentences is not None:
        messages.append(
            {
                "role": "user",
                "content": f"""Great. Now shorten the above summary to at most {num_sentences} sentences. Be sure to keep the most important information.

REMINDER: Continue to refer to each feature by its specific name. Only reference patterns and relationships explicitly present in the data. Do not fabricate or invent information not provided in the feature data.""",
            }
        )
        messages.append({"role": "assistant", "temperature": 0.3, "max_tokens": 2000})
    return messages