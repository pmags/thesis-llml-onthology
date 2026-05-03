"""Ontology schema definitions for the multi-level generation pipeline."""

from dataclasses import dataclass
from typing import Optional, List


def _pluralize_label(label: str) -> str:
    """Return a reasonable plural form for a level label."""
    lower_label = label.lower()
    if lower_label.endswith(("s", "x", "z", "ch", "sh")):
        return f"{label}es"
    if lower_label.endswith("y") and len(label) > 1 and lower_label[-2] not in "aeiou":
        return f"{label[:-1]}ies"
    return f"{label}s"


@dataclass
class OntologyLevel:
    """Defines one tier in the ontology hierarchy.

    Attributes:
        name: Level identifier used in DiGraph node attributes (e.g., "class").
        relation_to_parent: Edge relation label for edges FROM parent TO this level
            (e.g., "subClassOf"). None for the root level.
        rdf_predicate: The RDF predicate string to use when serializing edges
            from parent to this level (e.g., "rdfs:subClassOf", "rdf:type").
            None for the root level.
        is_rdf_class: Whether nodes at this level should get `rdf:type rdfs:Class`.
        expandable: Whether nodes at this level can be expanded (generate children).
            Leaf levels (e.g., instances) are not expandable.
        seed_key: The JSON key used in the seed taxonomy for this level's term name
            (e.g., "class" for classes/subclasses, "term" for instances).
        children_key: The JSON key used in the seed taxonomy to find children of this
            level (e.g., "subclasses", "instances"). None for leaf levels.
        plural_name: Optional plural label used in prompts (e.g., "classes").
        is_lexical: Whether this level represents lexical aliases/surface forms rather
            than domain ontology concepts. Lexical levels are useful for resolution,
            but should typically stay out of RDF serialization.
    """

    name: str
    relation_to_parent: Optional[str] = None
    rdf_predicate: Optional[str] = None
    is_rdf_class: bool = True
    expandable: bool = True
    seed_key: str = "class"
    children_key: Optional[str] = None
    plural_name: Optional[str] = None
    is_lexical: bool = False

    @property
    def pluralized_name(self) -> str:
        """Return the prompt-friendly plural label for this level."""
        return self.plural_name or _pluralize_label(self.name)


DEFAULT_LEVEL_SCHEMA: List[OntologyLevel] = [
    OntologyLevel(
        name="class",
        relation_to_parent=None,
        rdf_predicate=None,
        is_rdf_class=True,
        expandable=True,
        seed_key="class",
        children_key="subclasses",
        plural_name="classes",
    ),
    OntologyLevel(
        name="subclass",
        relation_to_parent="subClassOf",
        rdf_predicate="rdfs:subClassOf",
        is_rdf_class=True,
        expandable=True,
        seed_key="class",
        children_key="instances",
        plural_name="subclasses",
    ),
    OntologyLevel(
        name="instance",
        relation_to_parent="type",
        rdf_predicate="rdf:type",
        is_rdf_class=False,
        expandable=False,
        seed_key="term",
        children_key=None,
        plural_name="instances",
    ),
]
