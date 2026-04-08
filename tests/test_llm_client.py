"""Tests for provider-aware behavior in the ChatGpt client."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ontogen.llm_client import ChatGpt


class FakeStreamResponse:
    """Minimal streaming response stub for IAEDU client tests."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def raise_for_status(self) -> None:
        """Mimic a successful HTTP response."""

    def iter_lines(self, decode_unicode: bool = True):
        """Yield configured stream lines in order."""
        _ = decode_unicode
        for line in self._lines:
            yield line


class TestChatGptProviders:
    """Tests for provider-specific request handling."""

    def test_iaedu_defaults_to_serial_paced_requests(self):
        """IAEDU provider should default to conservative pacing settings."""
        fake_client = MagicMock()

        agent = ChatGpt(
            api_key="test-key",
            provider="iaedu",
            endpoint="https://api.iaedu.pt/agent-chat/example/stream",
            channel_id="channel-123",
            client=fake_client,
        )

        assert agent.min_request_interval_seconds == 1.0
        assert agent.max_concurrent_requests == 1

    def test_begin_request_waits_for_configured_interval(self):
        """Configured pacing should sleep before a request starts too soon."""
        fake_client = MagicMock()
        agent = ChatGpt(
            api_key="test-key",
            provider="iaedu",
            endpoint="https://api.iaedu.pt/agent-chat/example/stream",
            channel_id="channel-123",
            client=fake_client,
            min_request_interval_seconds=1.0,
            max_concurrent_requests=1,
        )
        agent._last_request_started_at = 10.0

        with patch("ontogen.llm_client.time.monotonic", return_value=10.2), patch(
            "ontogen.llm_client.time.sleep"
        ) as sleep_mock:
            request_state = agent._begin_request("phase-3-validation")
            agent._end_request(request_state)

        sleep_mock.assert_called_once()
        assert sleep_mock.call_args[0][0] == pytest.approx(0.8)

    def test_openai_chat_uses_responses_api(self):
        """OpenAI provider should call the Responses API and return output_text."""
        fake_responses = MagicMock()
        fake_responses.create.return_value = SimpleNamespace(
            output_text="hello")
        fake_client = SimpleNamespace(responses=fake_responses)

        agent = ChatGpt(
            model="gpt-4o-mini",
            api_key="test-key",
            provider="openai",
            client=fake_client,
        )

        result = agent.chat(
            instructions="You are helpful.",
            input="Say hello.",
        )

        assert result == "hello"
        fake_responses.create.assert_called_once_with(
            model="gpt-4o-mini",
            instructions="You are helpful.",
            input="Say hello.",
        )

    def test_endpoint_auto_selects_iaedu_provider(self):
        """Providing IAEDU endpoint details should switch the provider automatically."""
        fake_client = MagicMock()

        agent = ChatGpt(
            api_key="test-key",
            endpoint="https://api.iaedu.pt/agent-chat/example/stream",
            channel_id="channel-123",
            client=fake_client,
        )

        assert agent.provider == "iaedu"

    def test_iaedu_chat_returns_final_message_content(self):
        """IAEDU stream responses should be collapsed into one final string."""
        fake_client = MagicMock()
        fake_client.post.return_value = FakeStreamResponse(
            [
                '{"type": "start", "content": "Processing"}',
                '{"type": "token", "content": "IA"}',
                '{"type": "token", "content": "EDU"}',
                (
                    '{"type": "message", "content": '
                    '{"type": "ai", "content": "IAEDU_OK"}}'
                ),
                '{"type": "done", "content": "run-1"}',
            ]
        )

        agent = ChatGpt(
            api_key="test-key",
            provider="iaedu",
            endpoint="https://api.iaedu.pt/agent-chat/example/stream",
            channel_id="channel-123",
            client=fake_client,
        )

        result = agent.chat(
            instructions="Be concise.",
            input="Reply with exactly IAEDU_OK.",
        )

        assert result == "IAEDU_OK"
        assert agent.last_thread_id is not None
        assert len(agent.last_events) == 5

        _, kwargs = fake_client.post.call_args
        assert kwargs["stream"] is True
        assert kwargs["headers"]["x-api-key"] == "test-key"
        assert kwargs["files"]["channel_id"] == (None, "channel-123")

    def test_iaedu_error_event_raises_runtime_error(self):
        """IAEDU error-only streams should surface a clear RuntimeError."""
        fake_client = MagicMock()
        fake_client.post.return_value = FakeStreamResponse(
            [
                '{"type": "start", "content": "Processing"}',
                '{"type": "error", "content": "Unexpected processing error"}',
                '{"type": "done", "content": "run-1"}',
            ]
        )

        agent = ChatGpt(
            api_key="test-key",
            provider="iaedu",
            endpoint="https://api.iaedu.pt/agent-chat/example/stream",
            channel_id="channel-123",
            client=fake_client,
        )

        with pytest.raises(RuntimeError, match="Unexpected processing error"):
            agent.chat(
                instructions="Be concise.",
                input="Reply with exactly IAEDU_OK.",
            )


class TestChatGptJsonParsing:
    """Tests for response parsing shared across providers."""

    def test_similarity_parses_fenced_json_response(self):
        """Similarity parsing should tolerate JSON responses wrapped in code fences."""
        fake_responses = MagicMock()
        fake_responses.create.return_value = SimpleNamespace(
            output_text="unused")
        fake_client = SimpleNamespace(responses=fake_responses)

        agent = ChatGpt(
            model="gpt-4o-mini",
            api_key="test-key",
            provider="openai",
            client=fake_client,
        )
        agent.chat = MagicMock(
            return_value=(
                "```json\n"
                '{"term_x": "Spock", "description_x": "Half-Vulcan officer", '
                '"term_y": "Vulcans", "description_y": "Logical species", '
                '"similarity": 88}\n'
                "```"
            )
        )

        result = agent.get_similarity_with_descriptions(
            term_x="Spock",
            description_x="Half-Vulcan officer",
            term_y="Vulcans",
            description_y="Logical species",
        )

        assert result["similarity"] == 88

    def test_similarity_forwards_request_label_to_chat(self):
        """Similarity calls should forward the caller's request label for diagnostics."""
        fake_responses = MagicMock()
        fake_responses.create.return_value = SimpleNamespace(
            output_text="unused")
        fake_client = SimpleNamespace(responses=fake_responses)

        agent = ChatGpt(
            model="gpt-4o-mini",
            api_key="test-key",
            provider="openai",
            client=fake_client,
        )
        agent.chat = MagicMock(
            return_value=(
                '{"term_x": "Spock", "description_x": "Half-Vulcan officer", '
                '"term_y": "Vulcans", "description_y": "Logical species", '
                '"similarity": 88}'
            )
        )

        agent.get_similarity_with_descriptions(
            term_x="Spock",
            description_x="Half-Vulcan officer",
            term_y="Vulcans",
            description_y="Logical species",
            request_label="Phase 3 validation",
        )

        assert agent.chat.call_args.kwargs["request_label"] == "Phase 3 validation"
