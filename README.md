# TalkToEBM
[![Documentation](https://img.shields.io/badge/Documentation-View-blue)](http://interpret.ml/TalkToEBM/api_reference.html)
![python](https://img.shields.io/badge/python-3.8+-blue)
![Package Version](https://img.shields.io/pypi/v/t2ebm.svg?style=flat-square)
![License](https://img.shields.io/github/license/interpretml/TalkToEBM.svg?style=flat-square)
[![tests](https://github.com/interpretml/TalkToEBM/actions/workflows/run-pytest.yml/badge.svg)](https://github.com/interpretml/TalkToEBM/actions/workflows/run-pytest.yml)
[![Downloads](https://pepy.tech/badge/t2ebm)](https://pepy.tech/project/t2ebm)
<br/>

> ### A Natural Language Interface to Explainable Boosting Machines

<p align="center">
  <img src="images/landing.png" alt="drawing" width="900"/>
</p>

TalkToEBM is an open-source package that provides a natural language interface to [Explainable Boosting Machines (EBMs)](https://github.com/interpretml/interpret). With this package, you can convert the graphs of Explainable Boosting Machines to text and generate prompts for LLMs. We also have higher-level functions that directly ask the LLM to describe entire models. This package is under active development, so the current API is not guaranteed to stay stable.

Features:
- [x] Convert EBMs and their graphs to text that can be understood by LLMs.
- [x] Ask the LLM to describe and summarize individual graphs or entire models.
- [x] Modular approach that allows to write custom prompts - ask the LLM to perform any desired task with the EBM.
- [x] Automatic simplification of minor details in graphs to stay within the desired token limit.
- [x] Support for both OpenAI and local LLMs (like Ollama) for privacy and cost-sensitive applications.

# Installation

Python 3.8+ | Linux, Mac, Windows
```sh
pip install t2ebm
```

# High-Level API: Pass the EBM to the LLM

Here, we give an overview of high-level API functions that ask the LLM to describe graphs and EBMs. These functions take care of the prompts for you. The API Reference can be found [here](http://interpret.ml/TalkToEBM/api_reference.html).

First, we need to train a model. Here, we train an ```ExplainableBoostingClassifier``` on the Kaggle [Spaceship Titanic Dataset](https://www.kaggle.com/competitions/spaceship-titanic/overview).

```python
ebm = ExplainableBoostingClassifier(feature_names=feature_names)
ebm.fit(X_train, y_train)
```
We can now ask an LLM to describe a graph from the model. Asking the LLM to describe the graph for feature 0, 'Home Planet', is as simple as this:

```python
import t2ebm

t2ebm.describe_graph('gpt-4-turbo-2024-04-09', ebm, 0)
```

> **GPT-4:** *The graph illustrates the effects of the categorical feature `HomePlanet` on a
dependent variable, as modeled by a Generalized Additive Model (GAM). The
feature `HomePlanet` includes three categories: "Earth," "Europa," and "Mars."
Passengers from Europa exhibit a notably high positive mean effect of 0.5678 on
the dependent variable, with a tight confidence interval ranging from 0.5116 to
0.624, suggesting a strong and consistent positive impact. In contrast, Earth
shows a negative mean effect of -0.3246 with the confidence interval from -0.354
to -0.2952, indicating a robust negative association. Mars, while also positive,
has a much milder effect of 0.1713, with its confidence interval spanning from
0.1256 to 0.2171. This pronounced disparity in the effects, especially the
negative impact associated with Earth, is surprising and could hint at
underlying socio-economic or contextual factors influencing these outcomes.
Understanding these patterns might require further investigation into the
dataset's characteristics, including possible biases or the nature of the
dependent variable.*

Note that the LLM does not know about the dataset on which the model was trained. However, we can provide this information. This allows the LLM to talk about the outcome and to describe the graph in the context of the overall dataset. 

```python
dataset_description = """ The description of the dataset from Kaggle """

y_axis_descripton = """The y-axis depicts contributions in log-odds towards the probability
that the passenger was transported to another dimension."""
```


```python
t2ebm.describe_graph('gpt-4-turbo-2024-04-09', 
                      ebm,
                      0, 
                      graph_description=y_axis_descripton,
                      dataset_description=dataset_description)  
```

> **GPT-4:** *The graph from the Generalized Additive Model (GAM) showcases the contributions
of different home planets—Earth, Europa, and Mars—towards the log-odds of
passengers being transported to another dimension on the Spaceship Titanic.
Passengers from Europa exhibit a strong positive contribution to the likelihood
of being transported, with a mean log-odds contribution of 0.5678, indicating a
higher propensity for these individuals to experience the anomalous event.
Conversely, Earth shows a negative contribution (-0.3246 mean log-odds),
suggesting that passengers from Earth are less likely to be transported compared
to the baseline. Mars, while also showing a positive contribution (0.1713 mean
log-odds), has a lesser effect than Europa, but still indicates an increased
likelihood over Earth. This pattern could suggest unique environmental,
technological, or cultural factors on Europa and Mars that influence their
inhabitants' interactions with the spacetime anomaly encountered during the
voyage. The observed negative contribution from Earth might hint at more
effective protective measures or less vulnerability to such spacetime
disturbances. These findings highlight planet-based differences in
susceptibility to extraordinary events, revealing insights into interplanetary
travel and safety in this futuristic scenario.*

Instead of describing individual graphs, we can also ask the LLM to summarize the entire model.

```python
t2ebm.describe_ebm('gpt-4-turbo-2024-04-09',
                   ebm,
                   dataset_description=dataset_description, 
                   y_axis_description=y_axis_descripton) 
```

> **GPT-4:** *The Generalized Additive Model (GAM) used for analyzing the Spaceship Titanic
anomaly provides crucial insights into factors influencing the likelihood of
passengers being transported to an alternate dimension. Here's a concise summary
of the most impactful features:
>  1. **CryoSleep**: This feature significantly
affects the outcome, with passengers in cryosleep more likely to be transported
(mean effect size 0.814) compared to those who are not (mean effect -0.447).
This suggests a unique interaction between the cryosleep state and the anomaly,
potentially due to the location or conditions of cryosleep chambers.
>  2. **Spa**: Expenditures on spa services show a strong negative correlation with
the likelihood of transportation, particularly at higher spending levels. The
effect becomes extremely strong (below -5) at the highest expenditures,
indicating a protective factor potentially linked to socioeconomic status or
specific behaviors.
>  3. **VRDeck**: Similar to spa spending, expenditure on the
VRDeck is negatively correlated with the probability of transportation,
intensifying with higher spending. This suggests that engagement in VRDeck
amenities might be associated with safer areas or protective behaviors on the
ship.
>  4. **RoomService**: Initially, a slight increase in transportation
likelihood is observed at very low spending levels on room service, but it
shifts to a significant negative correlation as spending increases. High
expenditures on room service might correlate with safer locations on the ship.
>  5. **HomePlanet**: Passengers from Europa are much more likely to be transported
(mean effect 0.5678) compared to those from Earth (mean effect -0.3246) and Mars
(mean effect 0.1713). This indicates that planetary origin, reflecting differing
socio-economic or technological contexts, significantly influences
susceptibility to the anomaly.
>  6. **Cabin**: The cabin location, particularly
differences between Port and Starboard sides, significantly impacts the
likelihood of transportation. For instance, Starboard side cabins, especially on
specific decks (e.g., "C/S" with mean = 2.016), show higher positive effects.
>  7. **Destination**: The intended destination affects transportation likelihood,
with passengers destined for 55 Cancri e exhibiting a higher likelihood compared
to those heading to PSO J318.5-22 and TRAPPIST-1e. This might be influenced by
route or operational parameters specific to each destination.  The model
highlights the importance of understanding interactions between passenger states
(like cryosleep), cabin locations, spending on ship amenities, and origins in
assessing risks from spacetime anomalies. These factors play crucial roles in
the model's predictive accuracy and offer insights for enhancing safety and
design in future interstellar travel scenarios.*

# Low-Level API: Extract Graphs from the EBM and perform custom prompts

Most likely, you want the LLM to perform a custom task with the model. For example, you might want to ask the LLM to find [surprising and counter-intuitive patterns in the graphs](https://arxiv.org/abs/2308.01157). For this, we provide a low-level API that allows us to convert the graphs of EMBs to text. We also provide some basic prompt structures. You should use the functions described here if you want to write your own prompts based on the graphs of EBMs. The API Reference can be found [here](http://interpret.ml/TalkToEBM/api_reference.html).

```python
import t2ebm.graphs as graphs
```
We have a simple datastructure for graphs. Here, we extract a graph from an EBM and plot it:

```python
graph = graphs.extract_graph(ebm, 9)  # feature 9, 'Spa'
graphs.plot_graph(graph)
```

<img src="images/Spaceship%20Titanic_21_0.png" alt="drawing" width="400"/>

The graphs learned by EBMs can contain many small details. We can simplify them to reduce the number of tokens. There is a parameter to control the degree of simplification.

```python
t2ebm.graphs.plot_graph(t2ebm.graphs.simplify_graph(graph, min_variation_per_cent=0.041))
```

<img src="images/Spaceship%20Titanic_24_0.png" alt="drawing" width="400"/>


The package also supports categorical and boolean features

```python
graphs.plot_graph(graphs.extract_graph(ebm, 0)) # home planet
```

<img src="images/Spaceship%20Titanic_22_0.png" alt="drawing" width="400"/>

Before we can send a graph to an LLM, we need to convert it to text. The function ```graph_to_text``` takes care of this.

```python
graph_as_text = graphs.graph_to_text(graph, max_tokens=1000)
print(graph_as_text)
```

>    This graph represents a continuous-valued feature. The keys are intervals that represent ranges where the function predicts the same value.
>    
>    Feature Name: Spa
>    Feature Type: continuous
>    Means: {"(0.0, 36.5)": 0.517, "(36.5, 117.5)": 0.212, ..., "(3978.0, 18572.0)": -5.917}
> 
>    Lower Bounds (95%-Confidence Interval): {"(0.0, 36.5)": 0.068, "(36.5, 117.5)": 0.049, ..., "(3978.0, 18572.0)": -6.956}
> > 
>    Upper Bounds (95%-Confidence Interval): {"(0.0, 36.5)": 0.966, "(36.5, 117.5)": 0.374, ..., "(3978.0, 18572.0)": -4.877}
    
Given a graph's textual representation, we can embed it in a prompt. The function ```describe_graph``` provides a simple prompt template that can be used as a first step to build more complex prompts. 

```python
t2ebm.prompts.describe_graph(graph_as_text,
                             graph_description="{The y-axis on the graph depicts how cool the graph is.}",
                             dataset_description="{The dataset is great!}",
                             task_description="{Please summarize the graph. Pay special attention to run 'pip install t2ebm'}",)
```

```
Below is the graph of a Generalized Additive Model (GAM). The graph is presented as a JSON object with keys representing the x-axis and values representing the y-axis. For continuous features, the keys are intervals that represent ranges where the function predicts the same value. For categorical features, each key represents a possible value that the feature can take.
    
The graph is provided in the following format:
    - The name of the feature depicted in the graph
    - The type of the feature (continuous, categorical, or boolean)
    - Mean values
    - Lower bounds of confidence interval (optional)
    - Upper bounds of confidence interval (optional)

Here is the graph:

This graph represents categorical feature. Each key represents a possible value that the feature can take.

Feature Name: HomePlanet
Feature Type: categorical
Means: {"Earth": -0.3246, "Europa": 0.5678, "Mars": 0.1713}
Lower Bounds (95%-Confidence Interval): {"Earth": -0.354, "Europa": 0.5116, "Mars": 0.1256}
Upper Bounds (95%-Confidence Interval): {"Earth": -0.2952, "Europa": 0.624, "Mars": 0.2171}


{The y-axis on the graph depicts how cool the graph is.}

Here is a description of the dataset that the model was trained on:

{The dataset is great!}

{Please summarize the graph. Pay special attention to run 'pip install t2ebm'}
```

With this, you are ready to write your own prompts and ask the LLM to perform tasks with the graph. 

# Citation

If you use this software in your research, please consider citing our research papers.

```bib
@inproceedings{bordt2024talktoebm,
  author    = {Sebastian Bordt, Ben Lengerich, Harsha Nori, Rich Caruana},
  title     = {Data Science with LLMs and Interpretable Models},
  booktitle = {XAI4Sci Workshop at AAAI-24},
  year      = {2024}
 }
```

```bib
@inproceedings{lengerich2023llms,
  author    = {Benjamin J. Lengerich, Sebastian Bordt, Harsha Nori, Mark E. Nunnally, Yin Aphinyanaphongs, Manolis Kellis, and Rich Caruana},
  title     = {LLMs Understand Glass-Box Models, Discover Surprises, and Suggest Repairs},
  booktitle = {arxiv},
  year      = {2023}
 }
```

# Using Local LLMs (like Ollama)

TalkToEBM now supports local LLMs like Ollama as an alternative to OpenAI models. This allows you to use large language models locally without sending data to external APIs, which can be important for privacy, cost, or offline usage.

## Setting up Ollama

First, make sure you have Ollama installed and running. You can download Ollama from [https://ollama.ai/](https://ollama.ai/) and follow their installation instructions.

Once Ollama is running, you can pull a model like llama2 or minstral:

```sh
ollama pull llama2
```

## Using Local LLMs with TalkToEBM

To use a local LLM, you can specify a dictionary with the provider, model name, and base URL instead of an OpenAI model name:

```python
import t2ebm

# Define a local LLM configuration
local_llm = {
    "provider": "local",                 # Use "local"
    "model": "llama2",                   # Name of the model you've pulled into Ollama
    "base_url": "http://localhost:11434" # URL where Ollama is running
}

# Use the local LLM to describe a graph
t2ebm.describe_graph(local_llm, ebm, 0)
```

You can also use other local LLM servers that follow the OpenAI API format by using a generic configuration:

```python
# Example for a server following OpenAI API format
local_openai_compatible = {
    "provider": "local",
    "model": "your-model-name",
    "base_url": "http://localhost:8000"  # Your server address
}

t2ebm.describe_graph(local_openai_compatible, ebm, 0)
```

## Comparison with OpenAI

Local LLMs like those available through Ollama may not achieve the same level of sophistication as the latest OpenAI models, but they offer several advantages:

1. **Privacy**: Your data stays on your local machine
2. **No API costs**: Avoid per-token charges from commercial APIs
3. **Offline operation**: Work without an internet connection
4. **Customization**: Use specialized models fine-tuned for your domain

Remember that local models generally require more computing resources, especially RAM and a capable GPU for optimal performance.
