"""Server-side state management for the Dash ontology application."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, cast

from openai import APIStatusError
from requests import RequestException
from dotenv import load_dotenv

load_dotenv()

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

ONTOGEN_MODULE = importlib.import_module("ontogen")
ChatGpt = ONTOGEN_MODULE.ChatGpt
Ontology = ONTOGEN_MODULE.Ontology

APP_LOGGER = logging.getLogger("app.state")
LOG_BUFFER_SIZE = 400
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
_RUNTIME_LOG_STATE: Dict[str, Optional[Callable[[str], None]]] = {"sink": None}

DEFAULT_IAEDU_ENDPOINT = (
    "https://api.iaedu.pt/agent-chat//api/v1/agent/"
    "cmamvd3n40000c801qeacoad2/stream"
)
DEFAULT_IAEDU_CHANNEL_ID = "cmj1i57292iz9lq01goukbuuv"

APP_HANDLED_ERRORS = (
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
    OSError,
    RequestException,
    APIStatusError,
)

PLATEAU_DELTA = 0.02
PLATEAU_LIMIT = 5
STAGNATION_LIMIT = 8


class AppStateLogHandler(logging.Handler):
    """Forward formatted log records into the active app-state sink."""

    def emit(self, record: logging.LogRecord) -> None:
        """Send one log message to the configured in-memory sink."""
        log_sink = _RUNTIME_LOG_STATE["sink"]
        if log_sink is None:
            return

        message = self.format(record)
        _dispatch_log_line(cast(Callable[[str], None], log_sink), message)


def set_runtime_log_sink(log_sink: Callable[[str], None]) -> None:
    """Register the current in-memory sink used by the app log handler."""
    _RUNTIME_LOG_STATE["sink"] = log_sink


def _dispatch_log_line(log_sink: Callable[[str], None], message: str) -> None:
    """Invoke the active in-memory log sink."""
    log_sink(message)


def _resolve_log_level() -> int:
    """Resolve the configured app log level from the environment."""
    level_name = os.getenv("ONTOGEN_APP_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, level_name, logging.INFO)


def configure_runtime_logging() -> None:
    """Attach console and in-memory handlers for app and ontogen loggers."""
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    level = _resolve_log_level()

    for logger_name in ("app", "ontogen"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False

        if not any(handler.get_name() == f"{logger_name}-console" for handler in logger.handlers):
            console_handler = logging.StreamHandler(stream=sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.set_name(f"{logger_name}-console")
            logger.addHandler(console_handler)

        if not any(handler.get_name() == f"{logger_name}-buffer" for handler in logger.handlers):
            buffer_handler = AppStateLogHandler()
            buffer_handler.setFormatter(formatter)
            buffer_handler.set_name(f"{logger_name}-buffer")
            logger.addHandler(buffer_handler)


def _format_scope_description(
    scope_description: str,
    sub_domains: str,
) -> Optional[str]:
    """Build a single scope string from the free-text fields."""
    parts: List[str] = []
    if scope_description:
        parts.append(scope_description.strip())
    if sub_domains:
        normalized = ", ".join(
            part.strip() for part in sub_domains.split(",") if part.strip()
        )
        if normalized:
            parts.append(f"Sub-domains: {normalized}")
    if not parts:
        return None
    return " | ".join(parts)


def provider_is_ready(provider: str, api_key: Optional[str]) -> bool:
    """Return whether the selected provider has the required credentials."""
    provider_name = (provider or "iaedu").strip().lower()
    if provider_name == "openai":
        return bool(api_key or os.getenv("OPENAI_API_KEY"))
    return bool(api_key or os.getenv("IAEDU_API_KEY"))


def provider_status_message(provider: str, api_key: Optional[str]) -> str:
    """Return a user-facing readiness message for the selected provider."""
    provider_name = (provider or "iaedu").strip().lower()
    if provider_name == "openai":
        if api_key or os.getenv("OPENAI_API_KEY"):
            return "OpenAI key detected. The session can be initialized."
        return "Enter an OpenAI API key to enable initialization."

    if api_key or os.getenv("IAEDU_API_KEY"):
        return (
            "IAEDU key detected. Endpoint and channel will use the default notebook values "
            "unless workspace variables override them."
        )
    return (
        "Enter an IAEDU API key to initialize the session. Endpoint and channel use the "
        "default notebook values."
    )


@dataclass
class IterationLogEntry:
    """UI-friendly record for one generation iteration."""

    iteration: int
    node: Optional[str]
    generated: int
    accepted: int
    reward: float
    acceptance_rate: float
    nodes: int
    edges: int
    plateau_count: int
    stagnation_count: int
    elapsed_seconds: float
    status: str

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary for Dash tables."""
        return {
            "iteration": self.iteration,
            "node": self.node or "-",
            "generated": self.generated,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "reward": self.reward,
            "nodes": self.nodes,
            "edges": self.edges,
            "plateau": self.plateau_count,
            "stagnation": self.stagnation_count,
            "elapsed_seconds": self.elapsed_seconds,
            "status": self.status,
        }


@dataclass
class AppState:
    """Mutable in-memory state for a single-user Dash session."""

    ontology: Optional[Ontology] = None
    initialization_thread: Optional[threading.Thread] = None
    generation_thread: Optional[threading.Thread] = None
    generation_status: str = "idle"
    pause_event: threading.Event = field(default_factory=threading.Event)
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    log_lock: threading.Lock = field(default_factory=threading.Lock)
    iteration_log: List[IterationLogEntry] = field(default_factory=list)
    recent_logs: Deque[str] = field(default_factory=lambda: deque(maxlen=LOG_BUFFER_SIZE))
    config: Dict[str, Any] = field(default_factory=dict)
    seed_summary: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None
    last_message: str = "Awaiting ontology initialization."
    started_at_monotonic: Optional[float] = None
    plateau_count: int = 0
    stagnation_count: int = 0
    productive_rewards: List[float] = field(default_factory=list)
    previous_node_count: int = 0
    initialization_token: int = 0

    def reset(self) -> None:
        """Clear the active ontology and stop any background work."""
        self.stop_generation(wait=True)
        with self.lock:
            self.initialization_token += 1
            self.ontology = None
            self.initialization_thread = None
            self.generation_thread = None
            self.generation_status = "idle"
            self.pause_event.clear()
            self.stop_event.clear()
            self.iteration_log = []
            self.config = {}
            self.seed_summary = {}
            self.last_error = None
            self.last_message = "Awaiting ontology initialization."
            self.started_at_monotonic = None
            self.plateau_count = 0
            self.stagnation_count = 0
            self.productive_rewards = []
            self.previous_node_count = 0
        self.clear_logs()

    def has_ontology(self) -> bool:
        """Return True when an ontology has been initialized."""
        with self.lock:
            return self.ontology is not None

    def clear_logs(self) -> None:
        """Remove all buffered log lines for the current app session."""
        with self.log_lock:
            self.recent_logs.clear()

    def append_log_line(self, message: str) -> None:
        """Append one formatted log line to the in-memory session buffer."""
        with self.log_lock:
            self.recent_logs.append(message)

    def get_recent_logs(self) -> List[str]:
        """Return a copy of the buffered session log lines."""
        with self.log_lock:
            return list(self.recent_logs)

    def build_agent(
        self,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
        max_workers: int,
    ) -> ChatGpt:
        """Create a provider-aware ChatGpt client from UI inputs."""
        provider_name = (provider or "iaedu").strip().lower()
        if provider_name == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required when provider is OpenAI.")
            return ChatGpt(
                provider="openai",
                api_key=api_key,
                model=model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini",
                max_concurrent_requests=max_workers,
            )

        return ChatGpt(
            provider="iaedu",
            api_key=api_key or os.getenv("IAEDU_API_KEY"),
            endpoint=os.getenv("IAEDU_ENDPOINT") or DEFAULT_IAEDU_ENDPOINT,
            channel_id=os.getenv("IAEDU_CHANNEL_ID") or DEFAULT_IAEDU_CHANNEL_ID,
            max_concurrent_requests=1,
        )

    def _build_initialization_bundle(
        self,
        *,
        domain: str,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
        scope_description: str,
        sub_domains: str,
        exploration_constant: float,
        max_iterations: int,
        similarity_threshold: float,
        confidence_threshold: float,
        candidates_per_iteration: int,
        cross_link_threshold: float,
        retirement_limit: int,
        initial_seed_terms: int,
        max_workers: int,
    ) -> Dict[str, Any]:
        """Build the ontology seed and return the resulting session data."""
        clean_domain = domain.strip()
        if not clean_domain:
            raise ValueError("Domain is required.")

        agent = self.build_agent(provider, api_key, model, max_workers)
        scope = _format_scope_description(scope_description, sub_domains)
        ontology = Ontology(
            domain=clean_domain,
            scope_description=scope,
            agent=agent,
            exploration_constant=float(exploration_constant),
            max_iterations=int(max_iterations),
            similarity_threshold=float(similarity_threshold),
            confidence_threshold=float(confidence_threshold),
            candidates_per_iteration=int(candidates_per_iteration),
            cross_link_threshold=float(cross_link_threshold),
            retirement_limit=int(retirement_limit),
            initial_seed_terms=int(initial_seed_terms),
            max_workers=int(max_workers),
        )

        APP_LOGGER.info("Initializing ontology session for domain '%s'", clean_domain)
        seed = ontology.generate_initial_terms(num_classes=int(initial_seed_terms))
        if seed is None:
            raise RuntimeError(
                "Seed generation failed. The model returned an invalid taxonomy."
            )

        ontology.create_seed_ontology()
        validation_summary = ontology.validate_structure()

        model_name = model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
        top_level_count = len(seed.get("taxonomy", []))
        return {
            "ontology": ontology,
            "config": {
                "domain": clean_domain,
                "provider": provider,
                "model": model_name,
                "scope_description": scope_description,
                "sub_domains": sub_domains,
                "exploration_constant": float(exploration_constant),
                "max_iterations": int(max_iterations),
                "similarity_threshold": float(similarity_threshold),
                "confidence_threshold": float(confidence_threshold),
                "candidates_per_iteration": int(candidates_per_iteration),
                "cross_link_threshold": float(cross_link_threshold),
                "retirement_limit": int(retirement_limit),
                "initial_seed_terms": int(initial_seed_terms),
                "max_workers": int(max_workers),
            },
            "seed_summary": {
                "top_level_classes": top_level_count,
                "nodes": ontology.ontology_graph.number_of_nodes(),
                "edges": ontology.ontology_graph.number_of_edges(),
                "validation": validation_summary,
            },
            "message": (
                f"Initialized '{clean_domain}' with {top_level_count} top-level classes."
            ),
        }

    def _commit_initialization_bundle_locked(self, bundle: Dict[str, Any]) -> None:
        """Commit a completed initialization result while holding the state lock."""
        ontology = bundle["ontology"]
        self.ontology = ontology
        self.generation_thread = None
        self.generation_status = "ready"
        self.pause_event.clear()
        self.stop_event.clear()
        self.iteration_log = []
        self.config = bundle["config"]
        self.seed_summary = bundle["seed_summary"]
        self.last_error = None
        self.last_message = bundle["message"]
        self.started_at_monotonic = None
        self.plateau_count = 0
        self.stagnation_count = 0
        self.productive_rewards = []
        self.previous_node_count = ontology.ontology_graph.number_of_nodes()

    def start_initialization(
        self,
        *,
        domain: str,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
        scope_description: str,
        sub_domains: str,
        exploration_constant: float,
        max_iterations: int,
        similarity_threshold: float,
        confidence_threshold: float,
        candidates_per_iteration: int,
        cross_link_threshold: float,
        retirement_limit: int,
        initial_seed_terms: int,
        max_workers: int,
    ) -> str:
        """Start ontology initialization in the background so UI logs can stream live."""
        clean_domain = domain.strip()
        if not clean_domain:
            raise ValueError("Domain is required.")

        with self.lock:
            if self.initialization_thread is not None and self.initialization_thread.is_alive():
                raise RuntimeError("Ontology initialization is already running.")

        self.stop_generation(wait=True)
        self.clear_logs()
        set_runtime_log_sink(self.append_log_line)

        init_kwargs = {
            "domain": clean_domain,
            "provider": provider,
            "api_key": api_key,
            "model": model,
            "scope_description": scope_description,
            "sub_domains": sub_domains,
            "exploration_constant": exploration_constant,
            "max_iterations": max_iterations,
            "similarity_threshold": similarity_threshold,
            "confidence_threshold": confidence_threshold,
            "candidates_per_iteration": candidates_per_iteration,
            "cross_link_threshold": cross_link_threshold,
            "retirement_limit": retirement_limit,
            "initial_seed_terms": initial_seed_terms,
            "max_workers": max_workers,
        }

        with self.lock:
            self.initialization_token += 1
            token = self.initialization_token
            self.ontology = None
            self.initialization_thread = None
            self.generation_thread = None
            self.generation_status = "initializing"
            self.pause_event.clear()
            self.stop_event.clear()
            self.iteration_log = []
            self.config = {}
            self.seed_summary = {}
            self.last_error = None
            self.last_message = f"Initializing '{clean_domain}'. Live logs will appear below."
            self.started_at_monotonic = time.monotonic()
            self.plateau_count = 0
            self.stagnation_count = 0
            self.productive_rewards = []
            self.previous_node_count = 0

            thread = threading.Thread(
                target=self._initialization_loop,
                kwargs={"token": token, **init_kwargs},
                name="dash-ontology-initialization",
                daemon=True,
            )
            self.initialization_thread = thread

        APP_LOGGER.info("Starting ontology initialization for '%s'", clean_domain)
        thread.start()
        return "initializing"

    def _initialization_loop(
        self,
        *,
        token: int,
        domain: str,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
        scope_description: str,
        sub_domains: str,
        exploration_constant: float,
        max_iterations: int,
        similarity_threshold: float,
        confidence_threshold: float,
        candidates_per_iteration: int,
        cross_link_threshold: float,
        retirement_limit: int,
        initial_seed_terms: int,
        max_workers: int,
    ) -> None:
        """Run ontology initialization in a background thread."""
        try:
            bundle = self._build_initialization_bundle(
                domain=domain,
                provider=provider,
                api_key=api_key,
                model=model,
                scope_description=scope_description,
                sub_domains=sub_domains,
                exploration_constant=exploration_constant,
                max_iterations=max_iterations,
                similarity_threshold=similarity_threshold,
                confidence_threshold=confidence_threshold,
                candidates_per_iteration=candidates_per_iteration,
                cross_link_threshold=cross_link_threshold,
                retirement_limit=retirement_limit,
                initial_seed_terms=initial_seed_terms,
                max_workers=max_workers,
            )
        except APP_HANDLED_ERRORS as exc:
            APP_LOGGER.error("Ontology initialization failed: %s", exc)
            with self.lock:
                if token == self.initialization_token:
                    self.last_error = str(exc)
                    self.last_message = f"Initialization failed: {exc}"
                    self.generation_status = "error"
                    self.initialization_thread = None
                    self.started_at_monotonic = None
            return

        with self.lock:
            if token != self.initialization_token:
                return
            self._commit_initialization_bundle_locked(bundle)
            self.initialization_thread = None

        APP_LOGGER.info("Ontology initialization completed for '%s'", domain)

    def initialize_ontology(
        self,
        *,
        domain: str,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
        scope_description: str,
        sub_domains: str,
        exploration_constant: float,
        max_iterations: int,
        similarity_threshold: float,
        confidence_threshold: float,
        candidates_per_iteration: int,
        cross_link_threshold: float,
        retirement_limit: int,
        initial_seed_terms: int,
        max_workers: int,
    ) -> Dict[str, Any]:
        """Create an ontology, build its seed graph, and validate the initial structure."""
        set_runtime_log_sink(self.append_log_line)
        try:
            bundle = self._build_initialization_bundle(
                domain=domain,
                provider=provider,
                api_key=api_key,
                model=model,
                scope_description=scope_description,
                sub_domains=sub_domains,
                exploration_constant=exploration_constant,
                max_iterations=max_iterations,
                similarity_threshold=similarity_threshold,
                confidence_threshold=confidence_threshold,
                candidates_per_iteration=candidates_per_iteration,
                cross_link_threshold=cross_link_threshold,
                retirement_limit=retirement_limit,
                initial_seed_terms=initial_seed_terms,
                max_workers=max_workers,
            )
        except APP_HANDLED_ERRORS as exc:
            raise RuntimeError(str(exc)) from exc

        with self.lock:
            self.initialization_thread = None
            self.generation_thread = None
            self._commit_initialization_bundle_locked(bundle)
        return self.seed_summary.copy()

    def snapshot(self) -> Dict[str, Any]:
        """Return a read-only snapshot for UI callbacks."""
        recent_logs = self.get_recent_logs()
        with self.lock:
            ontology = self.ontology
            graph = ontology.ontology_graph if ontology is not None else None
            return {
                "has_ontology": ontology is not None,
                "domain": ontology.domain if ontology is not None else "",
                "generation_status": self.generation_status,
                "initialization_active": self.initialization_thread is not None
                and self.initialization_thread.is_alive(),
                "iteration_log": [entry.as_dict() for entry in self.iteration_log],
                "last_error": self.last_error,
                "last_message": self.last_message,
                "recent_logs": recent_logs,
                "config": self.config.copy(),
                "seed_summary": self.seed_summary.copy(),
                "mode": getattr(ontology, "expansion_mode", None) if ontology else None,
                "node_count": graph.number_of_nodes() if graph is not None else 0,
                "edge_count": graph.number_of_edges() if graph is not None else 0,
                "elapsed_seconds": self._elapsed_seconds(),
            }

    def get_expandable_nodes(self) -> List[str]:
        """Return currently expandable node ids."""
        with self.lock:
            if self.ontology is None:
                return []
            return [str(node) for node in self.ontology.list_expandable_nodes()]

    def expand_node(self, node: str) -> Dict[str, Any]:
        """Expand a single node manually and track the result for the UI."""
        set_runtime_log_sink(self.append_log_line)
        APP_LOGGER.info("Manual expansion requested for node '%s'", node)
        try:
            with self.lock:
                if self.ontology is None:
                    raise RuntimeError("No ontology has been initialized yet.")
                result = self.ontology.expand_node(node)
                entry = self._record_iteration_locked(result, status="manual")
                self.generation_status = "manual"
                self.last_error = None
                self.last_message = (
                    f"Expanded '{node}': {entry.accepted}/{entry.generated} accepted. "
                    "Automatic mode is no longer available for this ontology instance."
                )
                return {
                    "node": result.get("node"),
                    "candidates_generated": result.get("candidates_generated", 0),
                    "candidates_accepted": result.get("candidates_accepted", 0),
                    "reward": result.get("reward", 0.0),
                    "status": entry.status,
                }
        except APP_HANDLED_ERRORS as exc:
            raise RuntimeError(str(exc)) from exc

    def can_start_automatic(self) -> bool:
        """Return True when the current ontology instance still allows automatic mode."""
        with self.lock:
            if self.ontology is None:
                return False
            current_mode = getattr(self.ontology, "expansion_mode", None)
            return current_mode in (None, "automatic")

    def start_generation(self) -> str:
        """Start the background UCB1 generation loop if possible."""
        set_runtime_log_sink(self.append_log_line)
        with self.lock:
            if self.ontology is None:
                raise RuntimeError("No ontology has been initialized yet.")
            if getattr(self.ontology, "expansion_mode", None) == "manual":
                raise RuntimeError(
                    "Automatic generation is unavailable after manual expansion. "
                    "Create a new ontology to run UCB1 automation again."
                )
            if self.generation_thread is not None and self.generation_thread.is_alive():
                return self.generation_status

            self.stop_event.clear()
            self.pause_event.clear()
            self.generation_status = "running"
            self.last_error = None
            self.last_message = "Automatic generation started."
            if self.started_at_monotonic is None:
                self.started_at_monotonic = time.monotonic()
            thread = threading.Thread(
                target=self._generation_loop,
                name="dash-ontology-generation",
                daemon=True,
            )
            self.generation_thread = thread
            thread.start()
            APP_LOGGER.info("Automatic generation started")
            return self.generation_status

    def pause_generation(self) -> str:
        """Pause the background generation loop."""
        with self.lock:
            if self.generation_status != "running":
                return self.generation_status
            self.pause_event.set()
            self.generation_status = "paused"
            self.last_message = "Automatic generation paused."
            APP_LOGGER.info("Automatic generation paused")
            return self.generation_status

    def resume_generation(self) -> str:
        """Resume a paused background generation loop."""
        with self.lock:
            if self.generation_status != "paused":
                return self.generation_status
            self.pause_event.clear()
            self.generation_status = "running"
            self.last_message = "Automatic generation resumed."
            APP_LOGGER.info("Automatic generation resumed")
            return self.generation_status

    def stop_generation(self, wait: bool = False) -> str:
        """Request graceful termination of the background generation loop."""
        thread: Optional[threading.Thread]
        with self.lock:
            thread = self.generation_thread
            if thread is None:
                return self.generation_status
            self.stop_event.set()
            self.pause_event.clear()
            if self.generation_status not in {"completed", "idle"}:
                self.generation_status = "stopped"
                self.last_message = "Automatic generation stopped."
                APP_LOGGER.info("Automatic generation stop requested")

        if wait and thread.is_alive():
            thread.join(timeout=2)

        with self.lock:
            if self.generation_thread is thread and not thread.is_alive():
                self.generation_thread = None
        return self.generation_status

    def _generation_loop(self) -> None:
        """Run the automatic UCB1 loop and update state after each iteration."""
        while True:
            if self.stop_event.is_set():
                with self.lock:
                    self.generation_status = "stopped"
                    self.last_message = "Automatic generation stopped."
                    self.generation_thread = None
                return

            if self.pause_event.is_set():
                time.sleep(0.1)
                continue

            try:
                with self.lock:
                    ontology = self.ontology
                    if ontology is None:
                        raise RuntimeError("No ontology available for generation.")
                    if len(self.iteration_log) >= int(self.config.get("max_iterations", 0)):
                        self.generation_status = "completed"
                        self.last_message = "Reached the configured maximum iteration count."
                        self.generation_thread = None
                        return
                    result = ontology.expand_ontology()
                    entry = self._record_iteration_locked(result, status="running")
                    should_stop = self._should_stop_generation_locked(entry)
                    if should_stop:
                        self.generation_status = "completed"
                        self.generation_thread = None
                        return
            except APP_HANDLED_ERRORS as exc:
                with self.lock:
                    self.last_error = str(exc)
                    self.last_message = f"Generation failed: {exc}"
                    self.generation_status = "error"
                    self.generation_thread = None
                return

    def _record_iteration_locked(
        self,
        result: Dict[str, Any],
        *,
        status: str,
    ) -> IterationLogEntry:
        """Append one iteration entry while holding the state lock."""
        if self.ontology is None:
            raise RuntimeError("No ontology available.")

        generated = int(result.get("candidates_generated", 0) or 0)
        accepted = int(result.get("candidates_accepted", 0) or 0)
        reward = float(result.get("reward", 0.0) or 0.0)
        acceptance_rate = accepted / generated if generated else 0.0

        if accepted > 0:
            if self.productive_rewards:
                delta = abs(reward - self.productive_rewards[-1])
                if delta < PLATEAU_DELTA:
                    self.plateau_count += 1
                else:
                    self.plateau_count = 0
            self.productive_rewards.append(reward)

        current_node_count = self.ontology.ontology_graph.number_of_nodes()
        if current_node_count > self.previous_node_count:
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        self.previous_node_count = current_node_count

        entry = IterationLogEntry(
            iteration=len(self.iteration_log) + 1,
            node=result.get("node"),
            generated=generated,
            accepted=accepted,
            reward=reward,
            acceptance_rate=acceptance_rate,
            nodes=current_node_count,
            edges=self.ontology.ontology_graph.number_of_edges(),
            plateau_count=self.plateau_count,
            stagnation_count=self.stagnation_count,
            elapsed_seconds=self._elapsed_seconds(),
            status=status,
        )
        self.iteration_log.append(entry)
        self.last_error = None
        if result.get("node") is None:
            self.last_message = "No expandable nodes remain."
        else:
            self.last_message = (
                f"Iteration {entry.iteration}: expanded '{entry.node}' with "
                f"{entry.accepted}/{entry.generated} accepted candidates."
            )
        return entry

    def _should_stop_generation_locked(self, entry: IterationLogEntry) -> bool:
        """Apply app-level convergence and completion checks."""
        if entry.node is None:
            self.last_message = "No expandable nodes remain."
            return True
        if self.plateau_count >= PLATEAU_LIMIT:
            self.last_message = "Stopped after reward plateau detection."
            return True
        if self.stagnation_count >= STAGNATION_LIMIT:
            self.last_message = "Stopped after graph stagnation detection."
            return True
        return False

    def _elapsed_seconds(self) -> float:
        """Return elapsed generation time in seconds."""
        if self.started_at_monotonic is None:
            return 0.0
        return round(time.monotonic() - self.started_at_monotonic, 2)


APP_STATE = AppState()
configure_runtime_logging()
set_runtime_log_sink(APP_STATE.append_log_line)


def get_app_state() -> AppState:
    """Return the singleton app state instance."""
    return APP_STATE
