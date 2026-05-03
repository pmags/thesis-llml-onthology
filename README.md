# LLM-Driven Ontology Generation

Automatically generate hierarchical RDF/OWL ontologies from a user-specified domain using Large Language Models (LLMs). This system employs a **Top-Down Skeleton + Bottom-Up Population** algorithm that infers semantic relationships (similarity and composition) via LLM prompts, producing properly structured taxonomies with classes, subclasses, and instances.

**Example**: Given the domain "Star Trek", the system produces an ontology where `Species` (class) → `Vulcans` (subclass) → `Spock` (instance).

## Installation

### Prerequisites

- **Python ≥ 3.10**
- **Conda** (recommended) or pip
- **OpenAI API key** or **IAEDU agent credentials**

### Setup

1. **Clone and navigate to the repository:**
   ```bash
   git clone <repository-url>
   cd thesis-llml-onthology
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate thesis_env
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e '.[dev,notebooks,app]'
   ```

4. **Set your provider credentials:**
   ```bash
   export OPENAI_API_KEY=<your-api-key>

   # Optional IAEDU backend
   export IAEDU_API_KEY=<your-iaedu-api-key>
   export IAEDU_ENDPOINT=https://api.iaedu.pt/agent-chat//api/v1/agent/<agent-id>/stream
   export IAEDU_CHANNEL_ID=<your-channel-id>
   ```

## Quick Start

Generate an ontology for a domain in a few lines of code:

```python
from ontogen import ChatGpt, Ontology

# Initialize the OpenAI-backed LLM agent
agent = ChatGpt()

# Create and generate the ontology
onto = Ontology(domain="Star Trek", agent=agent)
onto.generate_ontology()
```

If you want an interactive app flow, the same `Ontology` instance can also
expand a user-selected node directly:

```python
from ontogen import ChatGpt, Ontology

agent = ChatGpt()
onto = Ontology(domain="Star Trek", agent=agent)

onto.seed = {
   "domain": "Star Trek",
   "taxonomy": [
      {
         "class": "Species",
         "description": "Sentient species",
         "subclasses": [],
      }
   ],
}
onto.create_seed_ontology()

expandable_nodes = onto.list_expandable_nodes(include_retired=True)
stats = onto.expand_node(expandable_nodes[0])
print(stats)
```

Manual expansion uses the same generation and validation pipeline as UCB1, but
it is a one-way choice per `Ontology` instance: after `expand_node()` is used,
that instance cannot switch back to automatic `generate_ontology()` or
`expand_ontology()`. Create a new `Ontology` instance if the user wants to go
back to automatic mode.

To use IAEDU with the same ontology code path:

```python
from ontogen import ChatGpt, Ontology

agent = ChatGpt(
   provider="iaedu",
   api_key="<your-iaedu-api-key>",
   endpoint="https://api.iaedu.pt/agent-chat//api/v1/agent/<agent-id>/stream",
   channel_id="<your-channel-id>",
)

onto = Ontology(domain="Star Trek", agent=agent)
onto.generate_ontology()
```

### Custom Ontology Schema

The ontology structure is configurable via the `level_schema` parameter. Override the default 3-level hierarchy (class → subclass → instance) if needed:

```python
from ontogen import Ontology, OntologyLevel

custom_levels = (
    OntologyLevel(
        name="category",
        relation_to_parent="categorizedBy",
        rdf_predicate="rdfs:subClassOf",
        is_rdf_class=True,
        expandable=True,
        seed_key="categories",
        children_key="items"
    ),
    # ... additional levels
)

onto = Ontology(domain="Star Trek", agent=agent, level_schema=custom_levels)
```

## Project Structure

```
src/ontogen/                 # Core package
├── __init__.py              # Public API exports
├── ontology.py              # Public facade and pipeline orchestrator
├── seed.py                  # Phase 1-2 seed generation and seed graph creation
├── validation.py            # Similarity cache and structural validation
├── expansion.py             # UCB1-guided iterative expansion
├── serialization.py         # RDF graph construction and serialization
├── visualization.py         # Notebook and HTML visualization helpers
├── progress.py              # Shared console progress formatting
├── llm_client.py            # ChatGpt LLM wrapper
├── models.py                # Ontology schema models
└── expansion_models.py      # Phase and iteration history records

app/                         # Gradio web UI
├── __init__.py
└── main.py

notebooks/
├── ontogen_sandbox.ipynb    # Main demo & experimentation
└── alternative_algo.ipynb   # Algorithm research

tests/                       # Pytest test suite
├── test_seed.py
├── test_similarity.py
├── test_validation.py
├── test_expansion.py
├── test_serialization.py
└── test_e2e.py

output/                      # Generated ontology files
├── ontology.ttl            # RDF/Turtle serialization

docs/
└── algorithm.md            # Algorithm documentation

environment.yml             # Conda environment spec
pyproject.toml              # Package configuration
README.md                   # This file
```

## Internal Architecture

`Ontology` remains the public entrypoint, but the internal implementation is split by responsibility through mixins so each phase is easier to navigate and debug.

- `SeedMixin` in `seed.py` handles schema helpers, prompt construction, and seed graph creation.
- `ValidationMixin` in `validation.py` handles similarity caching and parent-child structural pruning.
- `ExpansionMixin` in `expansion.py` handles both manual and UCB1-driven expansion, acceptance tracking, and convergence logic, including a revisit guard that prevents cold-start-only runs from being marked as plateau-converged.
- `SerializationMixin` in `serialization.py` handles RDF URI sanitization, triple creation, and serialization.
- `VisualizationMixin` in `visualization.py` handles notebook plots and interactive graph rendering.

This keeps the external API stable (`from ontogen import Ontology`) while allowing the phase implementations to evolve independently.

## Algorithm Overview

The generation process follows four phases:

1. **Structured Seed**: LLM generates a 3-level taxonomic skeleton (classes → subclasses → instances) as structured JSON.
2. **Validation**: Pairwise LLM similarity evaluation (~3n calls) pruning weak structural edges.
3. **Iterative Expansion**: Nodes can be expanded automatically via UCB1 or manually via `expand_node()`, both using the same candidate-generation and validation pipeline. Plateau convergence is only eligible after at least one actual revisit of an expandable node.
4. **RDF Serialization**: The ontology is mapped to standard RDF predicates (`rdfs:Class`, `rdfs:subClassOf`, `rdf:type`) and serialized to Turtle format.

For detailed algorithm documentation, see [docs/algorithm.md](docs/algorithm.md).

## Running Tests

Run the full test suite:

```bash
pytest tests/ -v
```

With coverage report:

```bash
pytest tests/ --cov=ontogen -v
```

## Web UI

Launch the Gradio web interface:

```bash
python -m app.main
```


## Configuration

Key parameters for `Ontology.__init__()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | `str` | Required | Domain for ontology generation (e.g., "Star Trek") |
| `agent` | `ChatGpt` | Required | LLM client for prompts |
| `level_schema` | `Tuple[OntologyLevel, ...]` | `DEFAULT_LEVEL_SCHEMA` | Customizable hierarchy levels |
| `num_classes` | `int` | 5 | Number of top-level classes in seed |
| `max_iterations` | `int` | 50 | Maximum expansion iterations |
| `candidates_per_iteration` | `int` | 3 | Candidates generated per expansion step |
| `similarity_threshold` | `float` | 0.5 | Acceptance threshold for candidates (0-1) |
| `cross_link_threshold` | `float` | 0.7 | Cross-branch linking threshold (0-1) |
| `exploration_constant` | `float` | 1.414 | UCB1 exploration parameter (√2) |
| `output_dir` | `Path` | `output/` | Directory for serialized ontologies |
| `namespace` | `str` | `http://example.org/ontology/` | RDF base namespace |

## License

This is a thesis project for research purposes.

## Additional Resources

- **Algorithm Documentation**: [docs/algorithm.md](docs/algorithm.md)
- **Main Notebook**: [notebooks/ontogen_sandbox.ipynb](notebooks/ontogen_sandbox.ipynb)
- **W3C RDF Schema**: https://www.w3.org/TR/rdf-schema/
