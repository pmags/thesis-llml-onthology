# Algorithm: Top-Down Skeleton + Bottom-Up Population

## 1. Overview

### The Approach

This ontology generation system uses a **Top-Down Skeleton + Bottom-Up Population** algorithm to transform a natural language domain description into a hierarchical RDF/OWL ontology. Rather than extracting flat lists of terms and computing expensive pairwise similarities (O(n²) LLM calls), the algorithm:

1. **Seeds** with a multi-level taxonomic skeleton (classes → subclasses → instances) via a single structured LLM prompt.
2. **Validates** the structure using ~3n targeted similarity checks (parent-child, sibling, cross-branch pairs).
3. **Expands** iteratively using a multi-armed bandit (UCB1) to intelligently select which taxonomy nodes to grow.
4. **Serializes** to RDF triples using W3C RDF Schema predicates.

### Why This Matters

The key insight is that **LLMs already understand hierarchies**—we just need to ask for them directly. This avoids:

- **O(n²) similarity calls** for pairwise comparisons (e.g., 100 terms = 5000 LLM calls at ~$0.01 each = $50).
- **Flat output** (all peers under one cluster label) by explicitly requesting multi-level structure.
- **Weak validation** by checking only the relationships that matter: direct parent-child edges, siblings, and cross-taxonomy links.

### Pipeline Diagram

```
Domain Input
    ↓
┌─────────────────────────────────────────┐
│ Phase 1: Structured Seed (~1 LLM call)  │
│ Generate multi-level taxonomy JSON      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Phase 2: Structural Validation (~3n)    │
│ Check parent-child, sibling,            │
│ cross-branch edges; prune weak ones    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Phase 3: UCB1 Expansion (~k×m calls)    │
│ Intelligently expand nodes;             │
│ validate & integrate new terms          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Phase 4: RDF Serialization (O(n))       │
│ Map DiGraph to RDF triples (Turtle)     │
└─────────────────────────────────────────┘
    ↓
RDF Output (ttl / xml)
```

---

## 2. RDF Schema Foundations

This algorithm produces ontologies conforming to [W3C RDF Schema (RDFS)](https://www.w3.org/TR/rdf-schema/). Three core RDFS predicates form the foundation:

### rdfs:Class
**Reference**: [RDFS §2.2](https://www.w3.org/TR/rdf-schema/#ch_class)

Declares a resource as a class (a category of things). In the ontology, **class nodes** are marked with `rdf:type rdfs:Class`.

```turtle
trekDomain:Species rdf:type rdfs:Class .
```

### rdfs:subClassOf
**Reference**: [RDFS §2.5](https://www.w3.org/TR/rdf-schema/#ch_subclassof)

Declares a hierarchical class-to-class relationship: "X is a subclass of Y" means all instances of X are also instances of Y.

```turtle
trekDomain:Vulcan rdfs:subClassOf trekDomain:Species .
```

### rdf:type
**Reference**: [RDFS §2.1](https://www.w3.org/TR/rdf-schema/#ch_type)

Declares an instance relationship: "X is of type Y" means X is a member of class Y.

```turtle
trekDomain:Spock rdf:type trekDomain:Vulcan .
```

### Default Predicate Schema

The algorithm uses these three predicates by default via `DEFAULT_LEVEL_SCHEMA`:

| Level | Name | RDF Predicate to Parent | Is Class? | Expandable? |
|-------|------|------------------------|-----------|------------|
| 0 | Class | *none (root)* | Yes | Yes |
| 1 | Subclass | `rdfs:subClassOf` | Yes | Yes |
| 2 | Instance | `rdf:type` | No | No |

This default maps naturally to the W3C RDF Schema specification. Custom hierarchies can be defined via the `OntologyLevel` dataclass, allowing projects to use different predicate names or deeper hierarchies.

---

## 3. Phase 1 — Structured Seed

### Why Structured Seed?

Instead of asking the LLM for "N random terms related to Star Trek," we ask for a **structured 3-level taxonomy**:

```json
{
  "domain": "Star Trek",
  "classes": [
    {
      "name": "Species",
      "subclasses": [
        {
          "name": "Vulcan",
          "instances": [
            { "name": "Spock", "description": "logical and analytical" },
            { "name": "Sarek", "description": "Spock's father" }
          ]
        },
        {
          "name": "Human",
          "instances": [
            { "name": "James Kirk", "description": "fearless captain" }
          ]
        }
      ]
    },
    {
      "name": "Role",
      "subclasses": [
        {
          "name": "Officer",
          "instances": [
            { "name": "James Kirk", "description": "starship captain" }
          ]
        }
      ]
    }
  ]
}
```

This structure directly maps to the hierarchical ontology we want to build—no flat term lists, no unstructured output.

### Implementation

**Method**: `Ontology.generate_initial_terms(num_classes: int = 5) → Optional[Dict[str, Any]]`

1. Build a schema-aware prompt from `self.level_schema`.
2. Send to LLM with request for `num_classes` root-level nodes.
3. Parse JSON defensively (handle extra text, missing keys, type errors).
4. Validate structure (non-empty classes, unique names).
5. Store in `self._seed_json` and return.

### Configurable Level Schema

The `OntologyLevel` dataclass allows custom hierarchies:

```python
@dataclass
class OntologyLevel:
    name: str                    # e.g., "Class", "Subclass", "Instance"
    relation_to_parent: str      # e.g., "subClassOf", "type"
    rdf_predicate: str           # e.g., "rdfs:subClassOf", "rdf:type"
    is_rdf_class: bool           # True if this level is an rdfs:Class
    expandable: bool             # True if nodes at this level can be expanded
    seed_key: str                # JSON key in seed response, e.g., "classes"
    children_key: str            # JSON key for children, e.g., "subclasses"
```

Users can define custom schemas:

```python
custom_schema = (
    OntologyLevel("Category", "", "rdfs:Class", True, True, "categories", "subcategories"),
    OntologyLevel("Subcategory", "subClassOf", "rdfs:subClassOf", True, True, "subcategories", "items"),
    OntologyLevel("Item", "type", "rdf:type", False, False, "items", ""),
)
onto = Ontology(domain="MyDomain", level_schema=custom_schema)
```

---

## 4. Phase 2 — Structural Validation

### Strategy: ~3n Validation Instead of n²

Building an RDF graph from the seed creates edges that may be weak or incorrect. Rather than validate all $\binom{n}{2}$ pairs, we validate ~3n targeted pairs:

**Parent-Child Pairs**: Every edge in the seed tree.
- Example: Is `Vulcan` really a subclass of `Species`?
- Expected to be strong (high similarity).

**Sibling Pairs**: Terms sharing the same parent.
- Example: Are `Vulcan` and `Klingon` both species?
- Expected to be moderate (they share a category).

**Cross-Branch Pairs**: Selected pairs from different top-level classes.
- Example: Is `Spock` (instance of Vulcan) also relevant to `Officer` (different class hierarchy)?
- Expected to be weak or moderate (supports cross-taxonomy typing).

### Similarity Evaluation

All pairs are evaluated using `_get_similarity_cached(term_a, term_b, description_a, description_b) → float`:

- Returns a float in [0, 100] representing semantic similarity.
- Cached by sorted tuple of terms to avoid redundant LLM calls.
- Calls either `ChatGpt.get_similarity()` or `ChatGpt.get_similarity_with_descriptions()` depending on whether descriptions are available.

### Pruning Rules

Weak edges are removed based on category-specific thresholds:

| Pair Category | Threshold | Interpretation |
|---------------|-----------|-----------------|
| **Parent-Child** | 50% | Child must be fairly distinct from parent (e.g., `Vulcan` vs `Species`). Too high → overly strict. Too low → weak children kept. |
| **Sibling** | 30% | Siblings can be moderately different; low threshold allows diversity within a category. |
| **Cross-Branch** | 70% | Cross-taxonomy links are rare; require high confidence. |

Edges below their threshold are removed from the graph. Nodes that become isolated (degree = 0) are flagged as orphaned and logged.

### Implementation

**Method**: `Ontology.validate_structure() → Dict[str, int]`

Returns a summary dictionary:

```python
{
    "parent_child_checked": 12,
    "parent_child_pruned": 2,
    "sibling_candidates": 8,
    "sibling_flagged": 1,
    "cross_branch_candidates": 3,
    "cross_branch_flagged": 1,
    "orphaned_nodes": 0,
}
```

---

## 5. Phase 3 — UCB1 Expansion

### Multi-Armed Bandit Formulation

After validating the seed, we **expand** the ontology by generating new subclasses and instances. But which nodes should we expand first?

The problem is a multi-armed bandit:

- **Arms** = nodes in the taxonomy (eligible for expansion).
- **Reward** = average similarity of newly accepted candidates under that node.
- **Goal** = balance exploration (try less-visited nodes) and exploitation (expand promising nodes).

### UCB1 Selection

The **Upper Confidence Bound (UCB1)** algorithm selects the node with the highest score:

$$\text{UCB1}_i = \bar{x}_i + c \sqrt{\frac{\ln N}{n_i}}$$

Where:

- $\bar{x}_i$ = mean reward for node $i$ (average of similarities of accepted candidates).
- $c$ = exploration constant (default: 1.41, $\sqrt{2}$).
- $N$ = total expansion iterations so far.
- $n_i$ = number of times node $i$ has been expanded.

**Intuition**:

- High $\bar{x}_i$ → node is "good" → exploit it.
- Low $n_i$ and high $N$ → node is "unexplored" → explore it.
- The $c$ parameter balances these forces.

### Algorithm Steps

1. **Select**: Use UCB1 to pick the next node to expand.
2. **Generate**: Prompt LLM for new subclasses (if node is a class) or instances (if subclass).
3. **Validate**: Check similarity of candidates to parent node (threshold: 50%).
4. **Integrate**: Add accepted candidates to the graph with `n_visits=0`, `total_reward=0.0`.
5. **Update**: Compute mean similarity of accepted candidates as reward; update the selected node's bandit stats.
6. **Repeat**: Until max iterations or plateau detected.

### Candidate Generation

**Method**: `_generate_candidates(node: str) → List[Dict[str, str]]`

Role-aware prompting based on node level:

- **For class nodes**: Generate new subclasses.
  - Prompt: "Generate 5 new subclasses of Species in the Star Trek universe. Include name and description."
  - Response: `[{"name": "...", "description": "..."}, ...]`

- **For subclass nodes**: Generate new instances.
  - Prompt: "Generate 5 new instances of Vulcan. Include name and description."
  - Response: Same JSON structure.

Context-aware: existing children are gathered and mentioned in the prompt to avoid duplicates.

### Candidate Validation & Integration

**Method**: `_validate_candidates(parent_node: str, candidates: List[Dict[str, str]]) → List[Dict[str, str]]`

Each candidate is evaluated via similarity to the parent node:

```
if similarity(candidate_name, parent_node) >= 50%:
    accept candidate
else:
    reject candidate
```

Accepted candidates are inserted into the graph with:

```python
graph.add_node(candidate_name, 
    level=parent_level + 1,
    n_visits=0,
    total_reward=0.0,
    description=candidate_description)
graph.add_edge(parent_node, candidate_name, 
    relation=relation_to_parent,
    relation_label=rdf_predicate)
```

### Early Termination

Expansion halts if:

- `max_iterations` (default: 50) is reached.
- **Plateau detected**: Recent iterations (last 5) show low average acceptance rates or low similarity scores, indicating the domain is saturated.

---

## 6. Phase 4 — RDF Serialization

### DiGraph to RDF Triples

The in-memory directed graph (DiGraph) is converted to RDF using `rdflib` and serialized to Turtle format.

**Mapping Rules**:

| DiGraph Entity | RDF Triple |
|----------------|-----------|
| Node with `level=class` (is_rdf_class=True) | `<node_uri> rdf:type rdfs:Class .` |
| Node with `level=subclass` OR `level=instance` where parent edge exists | `<child_uri> <relation_predicate> <parent_uri> .` |
| Edge with `relation_label=rdfs:subClassOf` | `<child> rdfs:subClassOf <parent> .` |
| Edge with `relation_label=rdf:type` | `<instance> rdf:type <class> .` |

### Example: Star Trek Ontology

Input graph:

```
Species (class)
├── Vulcan (subclass)
│   ├── Spock (instance)
│   └── Sarek (instance)
└── Human (class)
    └── Kirk (instance)

Officer (class)
└── Captain (subclass)
    └── Kirk (instance)
```

Output RDF (Turtle):

```turtle
@prefix trek: <http://example.org/startrek/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

trek:Species rdf:type rdfs:Class .
trek:Vulcan rdfs:subClassOf trek:Species .
trek:Spock rdf:type trek:Vulcan .
trek:Sarek rdf:type trek:Vulcan .

trek:Human rdfs:subClassOf trek:Species .
trek:Kirk rdf:type trek:Human .
trek:Kirk rdf:type trek:Captain .

trek:Officer rdf:type rdfs:Class .
trek:Captain rdfs:subClassOf trek:Officer .
```

Note: `Kirk` appears twice (cross-taxonomy typing: instance of both `Human` and `Captain`).

### URI Sanitization

Domain terms are converted to URIs using a sanitization function:

```python
def sanitize_uri(term: str) -> str:
    # CamelCase, remove spaces/punctuation, percent-encode special chars
    return urllib.parse.quote(term.replace(" ", "_"), safe="")
```

Example: `James Kirk` → `James_Kirk`; `Officer (Rank)` → `Officer_%28Rank%29`.

### Implementation

**Method**: `Ontology.serialize_ontology(output_path: str = "output/ontology.ttl") → str`

1. Create RDF namespace: `URIRef(f"http://example.org/{domain_name_sanitized}/")`.
2. Create RDF graph: `rdflib.Graph()`.
3. For each node in DiGraph:
   - If `is_rdf_class`: add triple `(node_uri, rdf:type, rdfs:Class)`.
   - For each outgoing edge: add triple based on edge `relation_label`.
4. Serialize to Turtle: `graph.serialize(output_path, format="turtle")`.
5. Return output path.

---

## 7. Complexity Analysis

### LLM Call Counts

| Phase | LLM Calls | Rationale |
|-------|-----------|-----------|
| **Seed** | O(1) | Single structured prompt returns entire skeleton. |
| **Validation** | O(n) | Check ~3 pairs per node (~3n total). Cached lookups avoid duplicates. |
| **Expansion** | O(k × m) | k = `max_iterations`, m = `candidates_per_iteration`. Each expansion generates m candidates and validates them. |
| **Serialization** | O(0) | No LLM calls; local graph traversal. |
| **Total** | O(n + k×m) | **Sub-quadratic** compared to O(n²) pairwise comparisons. |

### Practical Example: 50-Node Star Trek Ontology

Assume:
- Seed: 3 classes, ~15 total nodes after seed.
- Validation: ~45 pairs (3 × 15).
- Expansion: 50 iterations × 2 candidates/iteration = 100 LLM calls (to generate + validate).
- **Total: ~145 LLM calls** vs. **1225 pairs** for O(n²) approach.

**Cost**: ~$1.45 at $0.01/call vs. ~$12.25 for naive pairwise.

---

## 8. Configurable Level Schema

### The OntologyLevel Dataclass

All ontology structure is parameterized via `OntologyLevel` instances:

```python
@dataclass
class OntologyLevel:
    name: str                    # Display name, e.g., "Species"
    relation_to_parent: str      # Relation label in prompts, e.g., "subClassOf"
    rdf_predicate: str           # RDF predicate, e.g., "rdfs:subClassOf"
    is_rdf_class: bool           # Whether to emit rdf:type rdfs:Class
    expandable: bool             # Whether nodes can be selected for expansion
    seed_key: str                # JSON key in seed response, e.g., "classes"
    children_key: str            # JSON key for children, e.g., "subclasses"
```

### DEFAULT_LEVEL_SCHEMA

The default 3-level hierarchy:

```python
DEFAULT_LEVEL_SCHEMA = (
    OntologyLevel(
        name="Class",
        relation_to_parent="",
        rdf_predicate="rdfs:Class",
        is_rdf_class=True,
        expandable=True,
        seed_key="classes",
        children_key="subclasses",
    ),
    OntologyLevel(
        name="Subclass",
        relation_to_parent="subClassOf",
        rdf_predicate="rdfs:subClassOf",
        is_rdf_class=True,
        expandable=True,
        seed_key="subclasses",
        children_key="instances",
    ),
    OntologyLevel(
        name="Instance",
        relation_to_parent="type",
        rdf_predicate="rdf:type",
        is_rdf_class=False,
        expandable=False,
        seed_key="instances",
        children_key="",
    ),
)
```

### Custom Hierarchies

Users can define deep or specialized hierarchies:

```python
# 4-level hierarchy for a taxonomy-rich domain
custom_schema = (
    OntologyLevel("Kingdom", "", "rdfs:Class", True, True, "kingdoms", "phyla"),
    OntologyLevel("Phylum", "taxonomy/kingdom", "trek:hasPhylum", True, True, "phyla", "classes"),
    OntologyLevel("Class", "taxonomy/phylum", "trek:hasClass", True, True, "classes", "families"),
    OntologyLevel("Family", "taxonomy/class", "trek:hasFamily", True, False, "families", ""),
)

onto = Ontology(domain="Biology", level_schema=custom_schema)
```

All downstream phases automatically adjust:

- Seed generation prompts request levels in order.
- Validation checks relationships with correct predicates.
- Expansion role-awareness respects `expandable` flag.
- RDF serialization uses custom `rdf_predicate` values.

---

## References

- [W3C RDF Schema Specification](https://www.w3.org/TR/rdf-schema/)
- [W3C Semantic Web](https://www.w3.org/2001/sw/)
- Auer et al. (2007). "The Linked Open Data Cloud". Available online.
