"""Console progress helpers shared across ontology generation phases."""


def print_phase(phase_num: int, name: str) -> None:
    """Print a formatted phase header to stdout."""
    print(f"\n{'━' * 60}")
    print(f"  Phase {phase_num}: {name}")
    print(f"{'━' * 60}")
