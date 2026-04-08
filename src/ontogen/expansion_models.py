"""
    This module contains the Ontology class which orchestrates the full
ontology generation pipeline: seed generation, iterative expansion,
graph construction, and RDF serialization.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class ExpansionRecord:
    """Metrics captured for a single expansion iteration.

    Tracks what happened during one UCB1-guided expansion step, including
    which node was expanded, how many candidates were generated/accepted,
    the reward signal, and the cumulative graph state at that point.

    Attributes:
        iteration: 1-based iteration number.
        node_expanded: The node ID selected for expansion, or None if none available.
        candidates_generated: Number of candidate terms returned by the LLM.
        candidates_accepted: Number of candidates passing the similarity threshold.
        reward: Sum of accepted similarities divided by generated candidates (0–1 scale).
        cumulative_nodes: Total graph nodes after this iteration.
        cumulative_edges: Total graph edges after this iteration.
        acceptance_rate: Fraction of generated candidates that were accepted (0–1).
        plateau_count: Current consecutive plateau counter at this iteration.
        stagnation_count: Consecutive iterations where graph node count did not grow.
        elapsed_seconds: Wall-clock seconds since the pipeline started.
    """

    iteration: int
    node_expanded: Optional[str]
    candidates_generated: int
    candidates_accepted: int
    reward: float
    cumulative_nodes: int
    cumulative_edges: int
    acceptance_rate: float
    plateau_count: int
    stagnation_count: int
    elapsed_seconds: float


@dataclass
class PhaseRecord:
    """Timing and summary for a single pipeline phase.

    Attributes:
        phase: 1-based phase number.
        name: Human-readable phase name (e.g., "Seed generation").
        duration_seconds: Wall-clock duration of this phase.
        details: Arbitrary key-value details about what happened in the phase.
    """

    phase: int
    name: str
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationHistory:
    """Complete audit trail of an ontology generation run.

    Stores configuration, per-phase timing, per-iteration expansion metrics,
    and final summary statistics. Designed for post-hoc analysis, convergence
    plotting, and reproducibility auditing.

    Attributes:
        domain: The domain that was used for generation.
        started_at: ISO 8601 timestamp when the pipeline started.
        completed_at: ISO 8601 timestamp when the pipeline finished (None if still running).
        config: Snapshot of generation parameters (thresholds, max_iterations, etc.).
        phases: Ordered list of phase records with timing.
        expansion_records: Ordered list of per-iteration expansion metrics.
        total_iterations: Number of expansion iterations that ran.
        final_nodes: Final node count in the ontology graph.
        final_edges: Final edge count in the ontology graph.
        final_triples: Number of RDF triples in the serialized output.
        early_terminated: Whether the expansion loop stopped before max_iterations.
        termination_reason: Human-readable reason for termination.
    """

    domain: str
    started_at: str
    completed_at: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    phases: List[PhaseRecord] = field(default_factory=list)
    expansion_records: List[ExpansionRecord] = field(default_factory=list)
    total_iterations: int = 0
    final_nodes: int = 0
    final_edges: int = 0
    final_triples: int = 0
    early_terminated: bool = False
    termination_reason: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert expansion records to a pandas DataFrame for tabular analysis.

        Returns:
            DataFrame with one row per expansion iteration, columns matching
            ExpansionRecord fields.
        """
        if not self.expansion_records:
            return pd.DataFrame()

        return pd.DataFrame([asdict(r) for r in self.expansion_records])

    def summary(self) -> str:
        """Return a human-readable summary of the generation run.

        Returns:
            Multi-line string summarizing domain, timing, iterations,
            graph size, and termination status.
        """
        lines = [
            f"{'=' * 60}",
            "  Ontology Generation Summary",
            f"{'=' * 60}",
            f"  Domain:            {self.domain}",
            f"  Started:           {self.started_at}",
            f"  Completed:         {self.completed_at or 'N/A'}",
            f"  Iterations:        {self.total_iterations}",
            f"  Final nodes:       {self.final_nodes}",
            f"  Final edges:       {self.final_edges}",
            f"  RDF triples:       {self.final_triples}",
            f"  Early terminated:  {self.early_terminated}",
        ]
        if self.termination_reason:
            lines.append(f"  Reason:            {self.termination_reason}")

        # Phase timing breakdown
        if self.phases:
            lines.append(f"{'─' * 60}")
            lines.append("  Phase Timing:")
            for phase in self.phases:
                lines.append(
                    f"    {phase.phase}. {phase.name:<30s} {phase.duration_seconds:>7.1f}s")

        # Expansion summary statistics
        if self.expansion_records:
            rewards = [r.reward for r in self.expansion_records]
            accepted = [r.candidates_accepted for r in self.expansion_records]
            generated = [
                r.candidates_generated for r in self.expansion_records]
            lines.append(f"{'─' * 60}")
            lines.append("  Expansion Stats:")
            lines.append(
                f"    Avg reward:        {sum(rewards) / len(rewards):.3f}")
            lines.append(f"    Total generated:   {sum(generated)}")
            lines.append(f"    Total accepted:    {sum(accepted)}")
            total_gen = sum(generated)
            overall_rate = sum(accepted) / total_gen if total_gen > 0 else 0
            lines.append(f"    Overall accept %:  {overall_rate:.1%}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)
