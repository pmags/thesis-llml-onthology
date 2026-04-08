"""
LLM client wrapper for OpenAI and IAEDU interactions.

This module provides the ChatGpt class for sending prompts to either the
OpenAI Responses API or the IAEDU agent stream endpoint while keeping the same
public methods used by the ontology pipeline.
"""

import json
import logging
import os
import secrets
import threading
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

logger = logging.getLogger(__name__)

_DEFAULT_IAEDU_MIN_REQUEST_INTERVAL_SECONDS = 1.0
_DEFAULT_IAEDU_MAX_CONCURRENT_REQUESTS = 1
_DEFAULT_OPENAI_MAX_CONCURRENT_REQUESTS = 5
_SLOW_REQUEST_THRESHOLD_SECONDS = 5.0


class ChatGpt:
    """
    Provider-aware LLM client used by the ontology pipeline.

    The class keeps the existing ``chat()`` and similarity method contract so
    callers such as ``Ontology`` do not need to change when switching between
    providers.

    Supported providers:
    - ``openai``: uses the OpenAI Responses API.
    - ``iaedu``: uses the IAEDU multipart event-stream endpoint and collapses
      the stream into one final string.
    """

    SUPPORTED_PROVIDERS = frozenset({"openai", "iaedu"})

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        endpoint: Optional[str] = None,
        channel_id: Optional[str] = None,
        timeout: int = 60,
        min_request_interval_seconds: Optional[float] = None,
        max_concurrent_requests: Optional[int] = None,
        client: Optional[Any] = None,
        default_user_info: Optional[Dict[str, Any]] = None,
        default_user_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        load_dotenv()

        self.provider = self._resolve_provider(
            provider=provider,
            endpoint=endpoint,
            channel_id=channel_id,
        )
        self.model = model
        self.timeout = timeout
        self.endpoint = endpoint or self._get_endpoint_from_env()
        self.channel_id = channel_id or self._get_channel_id_from_env()
        self.min_request_interval_seconds = self._resolve_min_request_interval_seconds(
            min_request_interval_seconds
        )
        self.max_concurrent_requests = self._resolve_max_concurrent_requests(
            max_concurrent_requests
        )
        self.default_user_info = default_user_info or {}
        self.default_user_context = default_user_context or {}
        self.last_thread_id: Optional[str] = None
        self.last_events: List[Dict[str, Any]] = []
        self._request_gate_lock = threading.Lock()
        self._request_semaphore = threading.BoundedSemaphore(
            self.max_concurrent_requests
        )
        self._last_request_started_at = 0.0
        self._request_counter = 0

        self.api_key = api_key or self._get_api_key_from_env()
        self.client = client or self._build_client()

        self._validate_configuration()

        logger.info(
            "Initialized %s client (%s)",
            self.provider,
            self.describe_request_policy(),
        )

    def chat(
        self,
        instructions: str,
        prompt_text: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_info: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        request_label: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Send a chat request to the configured provider.

        Args:
            instructions: System-style instructions for the model.
            prompt_text: User input message to send.
            thread_id: Optional IAEDU thread id. Ignored by OpenAI.
            user_id: Optional IAEDU user id. Ignored by OpenAI.
            user_info: Optional IAEDU user info payload. Ignored by OpenAI.
            user_context: Optional IAEDU user context payload. Ignored by OpenAI.

        Returns:
            Final output text from the selected backend.
        """
        if prompt_text is None and "input" in kwargs:
            prompt_text = kwargs.pop("input")

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        if prompt_text is None:
            raise TypeError(
                "chat() missing required argument: 'input' or 'prompt_text'")

        if self.provider == "openai":
            return self._chat_openai(
                instructions=instructions,
                prompt_text=prompt_text,
                request_label=request_label,
            )

        return self._chat_iaedu(
            instructions=instructions,
            prompt_text=prompt_text,
            thread_id=thread_id,
            user_id=user_id,
            user_info=user_info,
            user_context=user_context,
            request_label=request_label,
        )

    def get_similarity(self, pairs_table: pd.DataFrame) -> pd.DataFrame:
        """
        Get similarity between pairs of terms using the configured provider.

        Uses a simple 0-10 scale prompt. This is the legacy method;
        prefer get_similarity_with_descriptions() for new code.

        Args:
            pairs_table: DataFrame with columns 'category_x' and 'category_y'
                representing pairs of terms.

        Returns:
            DataFrame with an additional column 'similarity' indicating the
            similarity score between the term pairs.
        """
        terms_pairs_json = pairs_table.to_json(orient="records")

        prompt_template = f"""
        In this survey you'll be asked to rate quantitatively, on a scale, the intensity of the
        semantic relatedness between pairs of affective words. Please, before starting, read
        carefully the instructions and the examples provided.

        The question we're asking is: how much related are the two words? Vaguely related words
        should be scored with lower values, and strongly related words with higher values. Please
        note that opposite words frequently present high values of relatedness.

        For example, the words 'modest' and 'smart' don't seem very related. 'Conceal' and 'mask'
        seem very related. 'Confident' is highly related with itself. 'Violent' and 'pacific',
        being opposite words, are frequently related, just like 'happiness' and 'sadness'.

        Examples:
        [
            {{"category_x":"modest","category_y":"smart","similarity":1}},
            {{"category_x":"conceal","category_y":"mask","similarity":7}},
            {{"category_x":"confident","category_y":"confident","similarity":10}},
            {{"category_x":"violent","category_y":"pacific","similarity":8}},
            {{"category_x":"happiness","category_y":"sadness","similarity":10}}
        ]

        Using the schema i provide and not encapsulate the output in code blocks,
        please rate from 0 to 10 the semantic relatedness of the following pairs of words,
        with 0 indicating words not related at all and 10 indicating very related words:
        {terms_pairs_json}
        """

        response_text = self.chat(
            instructions=(
                "You are a taxonomy and ontology expert. "
                "Provide concise and accurate responses based on the user's queries."
            ),
            input=prompt_template,
        )

        try:
            parsed = self._parse_json_response(response_text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Error parsing pair similarity response: %s", exc)
            parsed = []

        return pd.DataFrame(
            parsed,
            columns=["category_x", "category_y", "similarity"],
        )

    def get_similarity_with_descriptions(
        self,
        term_x: str,
        description_x: str,
        term_y: str,
        description_y: str,
        request_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get similarity between a single pair of concepts using the configured provider.

        Each concept consists of a term and its description. Returns a 0-100
        integer similarity score determined via prompt-based evaluation.

        Args:
            term_x: First term or concept name.
            description_x: Description of the first concept.
            term_y: Second term or concept name.
            description_y: Description of the second concept.

        Returns:
            Dictionary with keys 'term_x', 'description_x', 'term_y',
            'description_y', and 'similarity'.
        """
        prompt_template = f"""
        You are evaluating the semantic relatedness between pairs of concepts. Each concept
        consists of a TERM and its DESCRIPTION within a specific domain.

        TASK: Rate the intensity of the semantic relatedness between concept pairs on a scale
        from 0 to 100, where:
        - 0 = Not related at all
        - 100 = Highly related (including opposites, synonyms, or very closely connected concepts)

        IMPORTANT GUIDELINES:
        1. Consider BOTH the term and its description when evaluating relatedness
        2. Vaguely related concepts should receive lower scores (0-30)
        3. Moderately related concepts should receive middle scores (40-60)
        4. Strongly related concepts should receive higher scores (70-100)
        5. Note that OPPOSITE concepts often have HIGH relatedness scores because they are
           semantically connected (e.g., "hot" and "cold" are highly related as temperature
           opposites)
        6. Use any integer value between 0 and 100, not just multiples of 10

        EXAMPLES WITH EXPLANATIONS:

        Example 1 - Low Relatedness (Score: 10):
        {{
        "term_x": "modest",
        "description_x": "Having or showing a moderate estimation of one's own abilities",
        "term_y": "smart",
        "description_y": "Having or showing intelligence and mental capability",
        "similarity": 10
        }}

        Example 2 - High Relatedness (Score: 90):
        {{
        "term_x": "conceal",
        "description_x": "To hide something or prevent it from being known or seen",
        "term_y": "mask",
        "description_y": "To disguise or cover something to prevent identification",
        "similarity": 90
        }}

        Example 3 - Perfect Relatedness (Score: 100):
        {{
        "term_x": "confident",
        "description_x": "Feeling or showing certainty about something; self-assured",
        "term_y": "confident",
        "description_y": "Feeling or showing certainty about something; self-assured",
        "similarity": 100
        }}

        CONCEPT PAIR TO EVALUATE:
        {{
            "term_x": "{term_x}",
            "description_x": "{description_x}",
            "term_y": "{term_y}",
            "description_y": "{description_y}"
        }}

        RESPONSE FORMAT:
        Return ONLY a JSON object with this exact structure. Do NOT encapsulate in code blocks
        or markdown:
        {{
            "term_x": "{term_x}",
            "description_x": "{description_x}",
            "term_y": "{term_y}",
            "description_y": "{description_y}",
            "similarity": <integer between 0 and 100>
        }}
        """

        response_text = self.chat(
            instructions=(
                "You are a taxonomy and ontology expert. "
                "Provide concise and accurate responses based on the user's queries."
            ),
            input=prompt_template,
            request_label=request_label or f"similarity:{term_x}<->{term_y}",
        )

        try:
            result = self._parse_json_response(response_text)
            similarity = result.get("similarity")

            if isinstance(similarity, float):
                result["similarity"] = int(similarity)

            if not isinstance(result.get("similarity"), int) or not (
                0 <= result["similarity"] <= 100
            ):
                raise ValueError(
                    f"Invalid similarity score: {result.get('similarity')}"
                )

            return result
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "Error parsing response for pair (%s, %s): %s",
                term_x,
                term_y,
                exc,
            )
            return {
                "term_x": term_x,
                "description_x": description_x,
                "term_y": term_y,
                "description_y": description_y,
                "similarity": None,
            }

    def get_similarity_with_descriptions_batch(
        self,
        pairs: List[Dict[str, str]],
        request_label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple term-description pairs in a single LLM call.

        Args:
            pairs: List of dicts with keys 'term_x', 'description_x',
                'term_y', and 'description_y'.

        Returns:
            List of dicts (same order) with added key 'similarity' as an
            integer from 0 to 100.
        """
        if not pairs:
            return []

        pairs_json = json.dumps(pairs, ensure_ascii=False)

        prompt = f"""You are evaluating the semantic relatedness between multiple pairs of concepts.
Each concept consists of a TERM and its DESCRIPTION within a specific domain.

TASK: For each input pair, rate the intensity of semantic relatedness on a scale from 0 to 100
(integer):
- 0 = Not related at all
- 100 = Highly related (including opposites, synonyms, or very closely connected concepts)

IMPORTANT GUIDELINES:
1. Consider BOTH the term and its description when evaluating relatedness.
2. Vaguely related concepts should receive lower scores (0-30).
3. Moderately related concepts should receive middle scores (40-60).
4. Strongly related concepts should receive higher scores (70-100).
5. Opposite concepts can be highly related (e.g., "hot" vs "cold").
6. Use an integer value between 0 and 100.

CONCEPT PAIRS TO EVALUATE (preserve order):
{pairs_json}

RESPONSE FORMAT:
- Return ONLY a JSON ARRAY (no markdown, no code fences)
- Preserve the INPUT ORDER in your output
- Each item MUST contain exactly the keys: 'term_x','description_x','term_y',
  'description_y','similarity'
- 'similarity' must be an integer between 0 and 100
"""

        response_text = self.chat(
            instructions=(
                "You are a taxonomy and ontology expert. "
                "Provide concise JSON-only output in the same format as the inputs."
            ),
            input=prompt,
            request_label=request_label or f"batch-similarity:{len(pairs)}-pairs",
        )

        try:
            result = self._parse_json_response(response_text)
            if isinstance(result, list):
                for item in result:
                    similarity = item.get("similarity")
                    if isinstance(similarity, float):
                        item["similarity"] = int(similarity)
                return result
            raise ValueError("batch response is not a list")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Error parsing batch similarity response: %s", exc)
            return [
                {
                    "term_x": pair.get("term_x"),
                    "description_x": pair.get("description_x"),
                    "term_y": pair.get("term_y"),
                    "description_y": pair.get("description_y"),
                    "similarity": None,
                }
                for pair in pairs
            ]

    def generate_cluster_name(self, terms: List[str]) -> str:
        """
        Generate a concise descriptive name for a cluster of related terms.

        Args:
            terms: A list of terms that share a common theme or concept.

        Returns:
            A single-word or short-phrase name that best describes the common
            theme among the provided terms.
        """
        prompt_template = f"""
        You are a helpful assistant that generates concise and descriptive names for clusters
        of related terms. Given the following list of terms, provide a single-word or
        short-phrase name that best describes the common theme or concept among them.
        Do not encapsulate the output in code blocks, bold or any markdown.

        Terms: {terms}
        """

        response = self.chat(
            instructions=(
                "You are a taxonomy and ontology expert. "
                "Provide concise and accurate responses based on the user's queries."
            ),
            input=prompt_template,
        )
        return response.strip()

    def _build_client(self) -> Any:
        """Create the concrete backend client for the selected provider."""
        if self.provider == "openai":
            return OpenAI(api_key=self.api_key)

        return requests.Session()

    def describe_request_policy(self) -> str:
        """Return the active request pacing settings for diagnostics."""
        return (
            "provider=%s, max_concurrent_requests=%d, "
            "min_request_interval_seconds=%.2f"
        ) % (
            self.provider,
            self.max_concurrent_requests,
            self.min_request_interval_seconds,
        )

    def _begin_request(self, request_label: Optional[str] = None) -> Dict[str, Any]:
        """Acquire pacing gates before a provider request is sent."""
        label = request_label or "chat"

        slot_wait_start = time.monotonic()
        self._request_semaphore.acquire()
        slot_wait_seconds = time.monotonic() - slot_wait_start

        with self._request_gate_lock:
            self._request_counter += 1
            request_id = self._request_counter
            throttle_wait_seconds = 0.0
            now = time.monotonic()

            if (
                self.min_request_interval_seconds > 0
                and self._last_request_started_at > 0
            ):
                elapsed = now - self._last_request_started_at
                if elapsed < self.min_request_interval_seconds:
                    throttle_wait_seconds = (
                        self.min_request_interval_seconds - elapsed
                    )
                    logger.info(
                        "[%s #%d] Throttling '%s' for %.2fs",
                        self.provider.upper(),
                        request_id,
                        label,
                        throttle_wait_seconds,
                    )
                    time.sleep(throttle_wait_seconds)
                    now = time.monotonic()

            self._last_request_started_at = now

        if slot_wait_seconds > 0.01:
            logger.info(
                "[%s #%d] '%s' waited %.2fs for an available request slot",
                self.provider.upper(),
                request_id,
                label,
                slot_wait_seconds,
            )

        return {
            "request_id": request_id,
            "label": label,
            "slot_wait_seconds": slot_wait_seconds,
            "throttle_wait_seconds": throttle_wait_seconds,
            "started_at": time.monotonic(),
        }

    def _end_request(
        self,
        request_state: Dict[str, Any],
        *,
        output_chars: int = 0,
        event_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Release pacing gates and log request timing diagnostics."""
        duration_seconds = time.monotonic() - request_state["started_at"]

        try:
            if error:
                logger.error(
                    "[%s #%d] '%s' failed after %.2fs "
                    "(slot_wait=%.2fs, throttle_wait=%.2fs, events=%d, output_chars=%d): %s",
                    self.provider.upper(),
                    request_state["request_id"],
                    request_state["label"],
                    duration_seconds,
                    request_state["slot_wait_seconds"],
                    request_state["throttle_wait_seconds"],
                    event_count,
                    output_chars,
                    error,
                )
            elif (
                self.provider == "iaedu"
                or duration_seconds >= _SLOW_REQUEST_THRESHOLD_SECONDS
                or request_state["slot_wait_seconds"] > 0.01
                or request_state["throttle_wait_seconds"] > 0
            ):
                logger.info(
                    "[%s #%d] '%s' completed in %.2fs "
                    "(slot_wait=%.2fs, throttle_wait=%.2fs, events=%d, output_chars=%d)",
                    self.provider.upper(),
                    request_state["request_id"],
                    request_state["label"],
                    duration_seconds,
                    request_state["slot_wait_seconds"],
                    request_state["throttle_wait_seconds"],
                    event_count,
                    output_chars,
                )
        finally:
            self._request_semaphore.release()

    def _chat_openai(
        self,
        instructions: str,
        prompt_text: str,
        request_label: Optional[str] = None,
    ) -> str:
        """Send a chat request through the OpenAI Responses API."""
        request_state = self._begin_request(request_label or "openai-chat")
        output_text = ""
        error_summary: Optional[str] = None

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=prompt_text,
            )
            output_text = response.output_text
            return output_text
        except APIStatusError as exc:
            error_summary = f"OpenAI API error {exc.status_code}: {exc.message}"
            raise
        finally:
            self._end_request(
                request_state,
                output_chars=len(output_text),
                error=error_summary,
            )

    def _chat_iaedu(
        self,
        instructions: str,
        prompt_text: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_info: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        request_label: Optional[str] = None,
    ) -> str:
        """Send a chat request through the IAEDU stream endpoint."""
        request_state = self._begin_request(request_label or "iaedu-chat")
        current_thread_id = thread_id or self._new_thread_id()
        self.last_thread_id = current_thread_id

        headers = {
            "x-api-key": self.api_key,
            "Accept": "text/event-stream, application/json, text/plain",
        }
        files = self._build_iaedu_files(
            instructions=instructions,
            prompt_text=prompt_text,
            thread_id=current_thread_id,
            user_id=user_id,
            user_info=user_info,
            user_context=user_context,
        )

        final_message = ""
        error_message = ""
        token_buffer: List[str] = []
        events: List[Dict[str, Any]] = []
        output_text = ""
        error_summary: Optional[str] = None

        try:
            with self.client.post(
                self.endpoint,
                headers=headers,
                files=files,
                stream=True,
                timeout=self.timeout,
            ) as response:
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError:
                    error_summary = (
                        f"IAEDU HTTP error {response.status_code}: {response.reason}"
                    )
                    raise

                for raw_line in response.iter_lines(decode_unicode=True):
                    if raw_line is None:
                        continue

                    event = self._parse_iaedu_stream_line(raw_line)
                    if event is None:
                        continue

                    events.append(event)
                    event_type = event.get("type", "")
                    content = event.get("content")

                    if event_type == "token":
                        token_buffer.append(self._coerce_text(content))
                    elif event_type == "message":
                        final_message = self._coerce_text(content)
                    elif event_type == "error":
                        error_message = self._coerce_text(content)

            output_text = (final_message or "".join(token_buffer)).strip()

            if error_message and not output_text:
                error_summary = f"IAEDU error: {error_message}"
                raise RuntimeError(error_summary)

            if error_message and output_text:
                logger.warning(
                    "[%s #%d] '%s' returned output with a stream error marker: %s",
                    self.provider.upper(),
                    request_state["request_id"],
                    request_state["label"],
                    error_message,
                )

            return output_text
        finally:
            self.last_events = events
            self._end_request(
                request_state,
                output_chars=len(output_text),
                event_count=len(events),
                error=error_summary,
            )

    def _build_iaedu_files(
        self,
        instructions: str,
        prompt_text: str,
        thread_id: str,
        user_id: Optional[str] = None,
        user_info: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, tuple[None, str]]:
        """Build the multipart payload expected by IAEDU."""
        form_values = {
            "channel_id": self.channel_id,
            "thread_id": thread_id,
            "user_info": json.dumps(user_info or self.default_user_info),
            "message": self._build_iaedu_message(
                instructions=instructions,
                prompt_text=prompt_text,
            ),
        }

        merged_context = user_context or self.default_user_context
        if merged_context:
            form_values["user_context"] = json.dumps(merged_context)

        if user_id:
            form_values["user_id"] = user_id

        return {key: (None, str(value)) for key, value in form_values.items()}

    def _build_iaedu_message(self, instructions: str, prompt_text: str) -> str:
        """Flatten instructions and input into the IAEDU message field."""
        instructions = instructions.strip()
        prompt_text = prompt_text.strip()

        if instructions and prompt_text:
            return f"{instructions}\n\n{prompt_text}"
        return instructions or prompt_text

    def _parse_iaedu_stream_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single IAEDU event-stream line into a dictionary."""
        cleaned = line.strip()
        if not cleaned:
            return None

        if cleaned.startswith("data:"):
            cleaned = cleaned[5:].strip()

        if cleaned in {"[DONE]", "DONE", "done"}:
            return {"type": "done", "content": cleaned}

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"type": "raw", "content": cleaned}

        if isinstance(payload, dict):
            return payload

        return {"type": "raw", "content": payload}

    def _coerce_text(self, value: Any) -> str:
        """Extract plain text from a nested response payload."""
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            return "".join(self._coerce_text(item) for item in value)

        if isinstance(value, dict):
            for key in ("content", "text", "message", "delta", "answer"):
                if key in value:
                    text = self._coerce_text(value[key])
                    if text:
                        return text

        return ""

    def _parse_json_response(self, response_text: str) -> Any:
        """Parse JSON from a model response, tolerating code fences and wrappers."""
        logger.debug(
            "Raw model response (first 500 chars): %s",
            response_text[:500],
        )
        return json.loads(self._extract_json_payload(response_text))

    def _extract_json_payload(self, response_text: str) -> str:
        """Extract the most likely JSON object or array from a text response."""
        cleaned = response_text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass

        object_start = cleaned.find("{")
        array_start = cleaned.find("[")
        starts = [index for index in (
            object_start, array_start) if index != -1]
        if not starts:
            return cleaned

        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end > start:
            return cleaned[start:end + 1].strip()

        return cleaned

    def _resolve_provider(
        self,
        provider: Optional[str],
        endpoint: Optional[str],
        channel_id: Optional[str],
    ) -> str:
        """Resolve which provider backend should be used."""
        if provider:
            normalized = provider.strip().lower()
        elif (
            endpoint
            or channel_id
            or os.getenv("IAEDU_ENDPOINT")
            or os.getenv("IAEDU_CHANNEL_ID")
        ):
            normalized = "iaedu"
        else:
            normalized = "openai"

        if normalized not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{normalized}'. Must be one of: "
                f"{', '.join(sorted(self.SUPPORTED_PROVIDERS))}"
            )

        return normalized

    def _validate_configuration(self) -> None:
        """Validate provider-specific configuration after initialization."""
        if not self.api_key:
            env_var = (
                "OPENAI_API_KEY" if self.provider == "openai" else "IAEDU_API_KEY"
            )
            raise ValueError(
                f"Missing API key for provider '{self.provider}'. "
                f"Set {env_var} or pass api_key explicitly."
            )

        if self.provider == "iaedu":
            if not self.endpoint:
                raise ValueError(
                    "Missing IAEDU endpoint. Set IAEDU_ENDPOINT or pass endpoint explicitly."
                )
            if not self.channel_id:
                raise ValueError(
                    "Missing IAEDU channel id. Set IAEDU_CHANNEL_ID or pass channel_id explicitly."
                )

    def _get_api_key_from_env(self) -> Optional[str]:
        """Fetch the provider-specific API key from environment variables."""
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        return os.getenv("IAEDU_API_KEY")

    def _get_endpoint_from_env(self) -> Optional[str]:
        """Fetch the IAEDU endpoint from environment variables."""
        return os.getenv("IAEDU_ENDPOINT")

    def _get_channel_id_from_env(self) -> Optional[str]:
        """Fetch the IAEDU channel id from environment variables."""
        return os.getenv("IAEDU_CHANNEL_ID")

    def _resolve_min_request_interval_seconds(
        self,
        min_request_interval_seconds: Optional[float],
    ) -> float:
        """Resolve the minimum gap between request start times."""
        if min_request_interval_seconds is not None:
            return max(0.0, float(min_request_interval_seconds))

        env_var = (
            "IAEDU_MIN_REQUEST_INTERVAL_SECONDS"
            if self.provider == "iaedu"
            else "OPENAI_MIN_REQUEST_INTERVAL_SECONDS"
        )
        raw_value = os.getenv(env_var)
        if raw_value is None:
            if self.provider == "iaedu":
                return _DEFAULT_IAEDU_MIN_REQUEST_INTERVAL_SECONDS
            return 0.0

        try:
            return max(0.0, float(raw_value))
        except ValueError as exc:
            raise ValueError(
                f"Invalid value for {env_var}: {raw_value}"
            ) from exc

    def _resolve_max_concurrent_requests(
        self,
        max_concurrent_requests: Optional[int],
    ) -> int:
        """Resolve the request concurrency cap for this provider."""
        if max_concurrent_requests is not None:
            resolved = int(max_concurrent_requests)
        else:
            env_var = (
                "IAEDU_MAX_CONCURRENT_REQUESTS"
                if self.provider == "iaedu"
                else "OPENAI_MAX_CONCURRENT_REQUESTS"
            )
            raw_value = os.getenv(env_var)
            if raw_value is None:
                resolved = (
                    _DEFAULT_IAEDU_MAX_CONCURRENT_REQUESTS
                    if self.provider == "iaedu"
                    else _DEFAULT_OPENAI_MAX_CONCURRENT_REQUESTS
                )
            else:
                try:
                    resolved = int(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid value for {env_var}: {raw_value}"
                    ) from exc

        if resolved <= 0:
            raise ValueError("max_concurrent_requests must be >= 1")

        return resolved

    def _new_thread_id(self) -> str:
        """Create a fresh IAEDU thread id for an otherwise stateless call."""
        return secrets.token_urlsafe(16)
