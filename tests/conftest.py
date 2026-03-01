"""Shared test fixtures for ontogen tests."""

import pytest
from unittest.mock import MagicMock

from ontogen.llm_client import ChatGpt


@pytest.fixture
def mock_agent() -> MagicMock:
    """
    Provide a mocked ChatGpt instance that avoids real API calls.

    Returns:
        A MagicMock spec'd to ChatGpt with default return values.
    """
    agent = MagicMock(spec=ChatGpt)
    agent.chat.return_value = "[]"
    agent.get_similarity_with_descriptions.return_value = {
        "term_x": "A",
        "description_x": "desc A",
        "term_y": "B",
        "description_y": "desc B",
        "similarity": 75,
    }
    return agent
