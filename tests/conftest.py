"""Shared test fixtures for ontogen tests."""

import importlib
import pytest
import sys
from unittest.mock import create_autospec
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

ONTOGEN_LLM_CLIENT = importlib.import_module("ontogen.llm_client")
ChatGpt = ONTOGEN_LLM_CLIENT.ChatGpt


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
