"""Shared test fixtures for ontogen tests."""

import pytest
from unittest.mock import create_autospec

from ontogen.llm_client import ChatGpt


@pytest.fixture
def mock_agent():
    """
    Provide a mocked ChatGpt instance that avoids real API calls.

    Uses create_autospec to enforce method signatures for better test fidelity.
    Individual tests can override return_value or side_effect as needed.

    Returns:
        A ChatGpt mock with enforced signatures and default return values.
    """
    agent = create_autospec(ChatGpt, instance=True)
    
    # Set default return values that work with the enforced signatures
    agent.chat.return_value = "[]"
    agent.get_similarity_with_descriptions.return_value = {
        "term_x": "A",
        "description_x": "desc A",
        "term_y": "B",
        "description_y": "desc B",
        "similarity": 75,
    }
    
    return agent
