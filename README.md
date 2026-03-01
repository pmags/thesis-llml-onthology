# LLM-Driven Ontology Generation

Automatically generate hierarchical RDF/OWL ontologies from a user-specified domain using Large Language Models (LLMs). This system employs a **Top-Down Skeleton + Bottom-Up Population** algorithm that infers semantic relationships (similarity and composition) via LLM prompts, producing properly structured taxonomies with classes, subclasses, and instances.

**Example**: Given the domain "Star Trek", the system produces an ontology where `Species` (class) → `Vulcans` (subclass) → `Spock` (instance).

## Installation

### Prerequisites

- **Python ≥ 3.10**
- **Conda** (recommended) or pip
- **OpenAI API key** (set as `OPENAI_API_KEY` environment variable)

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

4. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY=<your-api-key>
   ```

## Quick Start

Generate an ontology for a domain in a few lines of code:

```python
from ontogen import ChatGpt, Ontology

# Initialize the LLM agent
agent = ChatGpt()

# Create and generate the ontology
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
├── ontology.py              # Ontology class: seed, expand, build, serialize
├── llm_client.py            # ChatGpt LLM wrapper
└── clustering.py            # Graph building and analysis

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

## Algorithm Overview

The generation process follows four phases:

1. **Structured Seed**: LLM generates a 3-level taxonomic skeleton (classes → subclasses → instances) as structured JSON.
2. **Validation**: Pairwise LLM similarity evaluation (~3n calls) pruning weak structural edges.
3. **Iterative Expansion**: UCB1 multi-armed bandit selects nodes to expand, generating new subclasses/instances with validation and cross-branch linking.
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

Or directly with Gradio:

```bash
gradio app/main.py
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
