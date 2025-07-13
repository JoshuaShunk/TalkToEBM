"""
Test prompt functionality.
"""

import t2ebm.prompts as prompts


class TestPrompts:
    """Test prompt generation functions."""

    def test_describe_graph_prompt(self):
        """Test describe_graph prompt generation."""
        graph_text = "Feature Name: test\nFeature Type: continuous"

        prompt = prompts.describe_graph(graph_text)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert graph_text in prompt

    def test_describe_graph_prompt_with_descriptions(self):
        """Test describe_graph prompt with additional descriptions."""
        graph_text = "Feature Name: test\nFeature Type: continuous"
        graph_desc = "This is a test graph"
        dataset_desc = "This is a test dataset"
        task_desc = "This is a test task"

        prompt = prompts.describe_graph(
            graph_text,
            graph_description=graph_desc,
            dataset_description=dataset_desc,
            task_description=task_desc,
        )

        assert isinstance(prompt, str)
        assert graph_text in prompt
        assert graph_desc in prompt
        assert dataset_desc in prompt
        assert task_desc in prompt

    def test_describe_ebm_prompt(self):
        """Test describe_ebm prompt generation."""
        model_text = "Model summary text"

        prompt = prompts.describe_ebm(model_text)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert model_text in prompt

    def test_describe_ebm_prompt_with_descriptions(self):
        """Test describe_ebm prompt with additional descriptions."""
        model_text = "Model summary text"
        dataset_desc = "This is a test dataset"
        y_axis_desc = "This is the y-axis description"

        prompt = prompts.describe_ebm(
            model_text, dataset_description=dataset_desc, y_axis_description=y_axis_desc
        )

        assert isinstance(prompt, str)
        assert model_text in prompt
        assert dataset_desc in prompt
        assert y_axis_desc in prompt

    def test_empty_graph_text(self):
        """Test prompt generation with empty graph text."""
        prompt = prompts.describe_graph("")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_none_descriptions(self):
        """Test prompt generation with None descriptions."""
        graph_text = "Feature Name: test"

        prompt = prompts.describe_graph(
            graph_text,
            graph_description=None,
            dataset_description=None,
            task_description=None,
        )

        assert isinstance(prompt, str)
        assert graph_text in prompt
