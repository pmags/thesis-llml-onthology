"""Tests for the Dash app support modules."""

import logging
import networkx as nx
import sys
import threading
import time
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from app.components import manual_expansion_progress_panel
from app.cytoscape_utils import (
    build_node_details,
    EXPAND_CONTEXT_MENU,
    EXPAND_CONTEXT_MENU_ID,
    graph_to_cytoscape,
)
from app.state import (
    AppState,
    Ontology,
    configure_runtime_logging,
    get_app_state,
    provider_is_ready,
    provider_status_message,
    set_runtime_log_sink,
)


def _build_sample_ontology(mock_agent):
    """Create a small ontology graph for UI utility tests."""
    ontology = Ontology(domain="Star Trek", agent=mock_agent)
    ontology.ontology_graph = nx.DiGraph()
    ontology.ontology_graph.add_node(
        "Species",
        term="Species",
        description="Sentient species in the Star Trek universe",
        level="class",
        n_visits=2,
        total_reward=1.2,
    )
    ontology.ontology_graph.add_node(
        "Vulcans",
        term="Vulcans",
        description="Logical species from Vulcan",
        level="subclass",
        n_visits=1,
        total_reward=0.7,
    )
    ontology.ontology_graph.add_node(
        "Spock",
        term="Spock",
        description="Half-Vulcan Starfleet officer",
        level="instance",
        n_visits=0,
        total_reward=0.0,
    )
    ontology.ontology_graph.add_edge("Species", "Vulcans", relation="subClassOf")
    ontology.ontology_graph.add_edge("Vulcans", "Spock", relation="type")
    return ontology


def test_graph_to_cytoscape_maps_nodes_and_edges(mock_agent):
    """Verify the helper emits Cytoscape elements with level classes and relation labels."""
    ontology = _build_sample_ontology(mock_agent)

    elements = graph_to_cytoscape(ontology.ontology_graph, ["Species", "Vulcans"])

    node_elements = [item for item in elements if "source" not in item["data"]]
    edge_elements = [item for item in elements if "source" in item["data"]]

    assert len(node_elements) == 3
    assert len(edge_elements) == 2
    assert any(item["classes"] == "class" for item in node_elements)
    assert any(item["data"]["label"] == "rdfs:subClassOf" for item in edge_elements)
    assert any(item["data"]["label"] == "rdf:type" for item in edge_elements)


def test_build_node_details_reports_neighbors_and_rewards(mock_agent):
    """Verify side-panel details include neighbors and computed mean reward."""
    ontology = _build_sample_ontology(mock_agent)

    details = build_node_details(ontology.ontology_graph, "Vulcans", ["Vulcans"])

    assert details["term"] == "Vulcans"
    assert details["parents"] == ["Species"]
    assert details["children"] == ["Spock"]
    assert details["expandable"] is True
    assert details["mean_reward"] == 0.7


def test_expand_context_menu_uses_cytoscape_context_menu_contract():
    """Verify right-click graph expansion is exposed through Dash Cytoscape context menus."""
    assert EXPAND_CONTEXT_MENU == [
        {
            "id": EXPAND_CONTEXT_MENU_ID,
            "label": "Expand ontology from this node",
            "availableOn": ["node"],
        }
    ]


def test_app_state_expand_node_records_manual_iteration(mock_agent):
    """Verify manual expansion appends an iteration record and locks the state to manual mode."""
    ontology = _build_sample_ontology(mock_agent)

    def fake_expand_node(node):
        ontology.ontology_graph.add_node(
            "Romulans",
            term="Romulans",
            description="Cunning species related to Vulcans",
            level="instance",
            n_visits=0,
            total_reward=0.0,
        )
        ontology.ontology_graph.add_edge(node, "Romulans", relation="type")
        ontology.expansion_mode = "manual"
        return {
            "node": node,
            "candidates_generated": 2,
            "candidates_accepted": 1,
            "reward": 0.4,
        }

    ontology.expand_node = fake_expand_node

    state = AppState(
        ontology=ontology,
        generation_status="ready",
        config={"max_iterations": 10},
        previous_node_count=ontology.ontology_graph.number_of_nodes(),
    )

    result = state.expand_node("Vulcans")

    assert result["node"] == "Vulcans"
    assert state.generation_status == "manual"
    assert len(state.iteration_log) == 1
    assert state.iteration_log[0].accepted == 1
    assert state.iteration_log[0].nodes == 4


def test_app_state_start_manual_expansion_runs_in_background(mock_agent):
    """Verify manual graph expansion streams logs while running in the background."""
    ontology = _build_sample_ontology(mock_agent)
    expansion_started = threading.Event()
    finish_expansion = threading.Event()

    def fake_expand_node(node):
        logging.getLogger("ontogen.ontology").info("Expanding test node %s", node)
        expansion_started.set()
        finish_expansion.wait(timeout=2)
        ontology.ontology_graph.add_node(
            "Romulans",
            term="Romulans",
            description="Cunning species related to Vulcans",
            level="instance",
            n_visits=0,
            total_reward=0.0,
        )
        ontology.ontology_graph.add_edge(node, "Romulans", relation="type")
        ontology.expansion_mode = "manual"
        return {
            "node": node,
            "candidates_generated": 2,
            "candidates_accepted": 1,
            "reward": 0.4,
        }

    ontology.expand_node = fake_expand_node
    state = AppState(
        ontology=ontology,
        generation_status="ready",
        config={"max_iterations": 10},
        previous_node_count=ontology.ontology_graph.number_of_nodes(),
    )
    configure_runtime_logging()

    assert state.start_manual_expansion("Vulcans") == "manual_expanding"
    assert expansion_started.wait(timeout=2)

    running_snapshot = state.snapshot()
    assert running_snapshot["generation_status"] == "manual_expanding"
    assert running_snapshot["manual_expansion_active"] is True
    assert running_snapshot["manual_expansion_node"] == "Vulcans"
    assert any("Expanding test node Vulcans" in line for line in running_snapshot["recent_logs"])

    finish_expansion.set()
    timeout_at = time.time() + 2
    while time.time() < timeout_at:
        if not state.snapshot()["manual_expansion_active"]:
            break
        time.sleep(0.05)

    completed_snapshot = state.snapshot()
    assert completed_snapshot["generation_status"] == "manual"
    assert completed_snapshot["manual_expansion_active"] is False
    assert len(completed_snapshot["iteration_log"]) == 1
    assert completed_snapshot["iteration_log"][0]["node"] == "Vulcans"

    set_runtime_log_sink(get_app_state().append_log_line)


def test_manual_expansion_progress_panel_renders_logs():
    """Verify the bottom offcanvas body includes status metrics and recent logs."""
    panel = manual_expansion_progress_panel(
        {
            "manual_expansion_active": True,
            "manual_expansion_node": "Vulcans",
            "generation_status": "manual_expanding",
            "last_error": None,
            "last_message": "Expanding 'Vulcans'.",
            "recent_logs": ["log line one", "log line two"],
            "iteration_log": [],
            "node_count": 3,
            "elapsed_seconds": 1.25,
        }
    )

    assert panel.children[1].children == "Expanding from: Vulcans"
    assert panel.children[-1].children == "log line one\nlog line two"


def test_provider_is_ready_for_iaedu_with_inline_key(monkeypatch):
    """Verify IAEDU readiness only depends on an API key because endpoint and channel have defaults."""
    monkeypatch.delenv("IAEDU_API_KEY", raising=False)

    assert provider_is_ready("iaedu", "inline-key") is True
    assert "default notebook values" in provider_status_message("iaedu", "inline-key")


def test_app_state_captures_ontogen_logs_in_recent_buffer():
    """Verify ontogen logger messages are mirrored into the app session log buffer."""
    state = AppState()
    configure_runtime_logging()
    set_runtime_log_sink(state.append_log_line)

    logging.getLogger("ontogen.ontology").info("Phase 1: Generating seed from domain Test")

    snapshot = state.snapshot()
    assert any("Phase 1: Generating seed from domain Test" in line for line in snapshot["recent_logs"])

    set_runtime_log_sink(get_app_state().append_log_line)


def test_app_state_start_initialization_runs_in_background_and_collects_logs(mock_agent):
    """Verify initialization runs asynchronously and exposes ontogen logs in the state snapshot."""
    ontology = _build_sample_ontology(mock_agent)
    
    class BackgroundInitState(AppState):
        """Test double that returns a synthetic initialization result."""

        def _build_initialization_bundle(self, **kwargs):
            logging.getLogger("ontogen.ontology").info(
                "Phase 1: Generating seed from domain %s",
                kwargs["domain"],
            )
            return {
                "ontology": ontology,
                "config": {"domain": kwargs["domain"], "provider": kwargs["provider"]},
                "seed_summary": {
                    "top_level_classes": 1,
                    "nodes": ontology.ontology_graph.number_of_nodes(),
                    "edges": ontology.ontology_graph.number_of_edges(),
                    "validation": {"edges_pruned": 0, "orphaned_nodes": 0},
                },
                "message": f"Initialized '{kwargs['domain']}' with 1 top-level classes.",
            }

    state = BackgroundInitState()

    assert (
        state.start_initialization(
            domain="Astrophysics",
            provider="iaedu",
            api_key="inline-key",
            model=None,
            scope_description="",
            sub_domains="",
            exploration_constant=2.0,
            max_iterations=5,
            similarity_threshold=0.5,
            confidence_threshold=0.5,
            candidates_per_iteration=10,
            cross_link_threshold=70,
            retirement_limit=3,
            initial_seed_terms=5,
            max_workers=1,
        )
        == "initializing"
    )

    timeout_at = time.time() + 2
    while time.time() < timeout_at:
        thread = state.initialization_thread
        if thread is None or not thread.is_alive():
            break
        time.sleep(0.05)

    snapshot = state.snapshot()
    assert snapshot["generation_status"] == "ready"
    assert snapshot["has_ontology"] is True
    assert any("Phase 1: Generating seed from domain Astrophysics" in line for line in snapshot["recent_logs"])

    set_runtime_log_sink(get_app_state().append_log_line)


def test_app_state_start_run_can_auto_start_generation(mock_agent):
    """Verify start_run can chain background initialization into automatic generation."""
    ontology = _build_sample_ontology(mock_agent)
    start_generation_calls = {"value": 0}

    class AutoRunState(AppState):
        """Test double that makes automatic follow-up generation observable."""

        def _build_initialization_bundle(self, **kwargs):
            return {
                "ontology": ontology,
                "config": {"domain": kwargs["domain"], "max_iterations": 4},
                "seed_summary": {
                    "top_level_classes": 1,
                    "nodes": ontology.ontology_graph.number_of_nodes(),
                    "edges": ontology.ontology_graph.number_of_edges(),
                    "validation": {"edges_pruned": 0, "orphaned_nodes": 0},
                },
                "message": f"Initialized '{kwargs['domain']}' with 1 top-level classes.",
            }

        def start_generation(self):
            start_generation_calls["value"] += 1
            self.generation_status = "running"
            self.last_message = "Automatic generation started."
            return "running"

    state = AutoRunState()

    assert (
        state.start_run(
            domain="Astrophysics",
            provider="iaedu",
            api_key="inline-key",
            model=None,
            scope_description="",
            sub_domains="",
            exploration_constant=2.0,
            max_iterations=4,
            similarity_threshold=0.5,
            confidence_threshold=0.5,
            candidates_per_iteration=10,
            cross_link_threshold=70,
            retirement_limit=3,
            initial_seed_terms=5,
            max_workers=1,
            automatic=True,
        )
        == "initializing"
    )

    timeout_at = time.time() + 2
    while time.time() < timeout_at:
        thread = state.initialization_thread
        if thread is None or not thread.is_alive():
            break
        time.sleep(0.05)

    snapshot = state.snapshot()
    assert start_generation_calls["value"] == 1
    assert snapshot["generation_status"] == "running"
    assert snapshot["auto_start_generation"] is True


def test_app_state_stop_run_invalidates_active_run():
    """Verify stop_run marks an active ontology job as stopped."""
    state = AppState(
        generation_status="initializing",
        initialization_token=4,
        auto_start_generation=True,
    )

    assert state.stop_run() == "stopped"

    snapshot = state.snapshot()
    assert snapshot["generation_status"] == "stopped"
    assert snapshot["auto_start_generation"] is False
    assert state.initialization_token == 5
    assert state.stop_event.is_set()


def test_app_state_start_generation_completes_background_loop(mock_agent):
    """Verify the automation loop runs in the background and records iterations."""
    ontology = _build_sample_ontology(mock_agent)
    call_count = {"value": 0}

    def fake_expand_ontology():
        if call_count["value"] == 0:
            ontology.ontology_graph.add_node(
                "Romulans",
                term="Romulans",
                description="Related species from Romulus",
                level="instance",
                n_visits=0,
                total_reward=0.0,
            )
            ontology.ontology_graph.add_edge("Vulcans", "Romulans", relation="type")
            call_count["value"] += 1
            return {
                "node": "Vulcans",
                "candidates_generated": 2,
                "candidates_accepted": 1,
                "reward": 0.5,
            }

        call_count["value"] += 1
        return {
            "node": None,
            "candidates_generated": 0,
            "candidates_accepted": 0,
            "reward": 0.0,
        }

    ontology.expand_ontology = fake_expand_ontology

    state = AppState(
        ontology=ontology,
        generation_status="ready",
        config={"max_iterations": 4},
        previous_node_count=ontology.ontology_graph.number_of_nodes(),
    )

    assert state.start_generation() == "running"

    timeout_at = time.time() + 2
    while time.time() < timeout_at:
        thread = state.generation_thread
        if thread is None or not thread.is_alive():
            break
        time.sleep(0.05)

    snapshot = state.snapshot()
    assert snapshot["generation_status"] == "completed"
    assert len(snapshot["iteration_log"]) == 2
    assert snapshot["iteration_log"][0]["node"] == "Vulcans"


def test_app_state_elapsed_freezes_after_completion(mock_agent):
    """Verify completed ontology runs keep a stable elapsed value."""
    ontology = _build_sample_ontology(mock_agent)
    state = AppState(
        ontology=ontology,
        generation_status="ready",
        config={"max_iterations": 0},
        started_at_monotonic=time.monotonic() - 5.0,
    )

    assert state.start_generation() == "running"

    timeout_at = time.time() + 2
    while time.time() < timeout_at:
        thread = state.generation_thread
        if thread is None or not thread.is_alive():
            break
        time.sleep(0.05)

    first_snapshot = state.snapshot()
    time.sleep(0.05)
    second_snapshot = state.snapshot()

    assert first_snapshot["generation_status"] == "completed"
    assert first_snapshot["elapsed_seconds"] == second_snapshot["elapsed_seconds"]
