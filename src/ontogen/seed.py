"""Seed-generation and seed-graph construction helpers for the ontology pipeline."""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from ontogen.models import OntologyLevel

if TYPE_CHECKING:
    from ontogen.ontology import Ontology


logger = logging.getLogger("ontogen.ontology")


class SeedMixin:
    """Mixin containing schema helpers and Phase 1–2 implementation details."""

    def _get_level(self: "Ontology", name: str) -> OntologyLevel:
        """Retrieve a level definition by name."""
        for level in self.level_schema:
            if level.name == name:
                return level
        raise ValueError(f"Level '{name}' not found in level_schema")

    def _get_child_level(
        self: "Ontology",
        parent_level_name: str,
    ) -> Optional[OntologyLevel]:
        """Get the next level down in the hierarchy."""
        parent_index = None
        for index, level in enumerate(self.level_schema):
            if level.name == parent_level_name:
                parent_index = index
                break

        if parent_index is None:
            raise ValueError(
                f"Parent level '{parent_level_name}' not found in level_schema"
            )

        if parent_index + 1 < len(self.level_schema):
            return self.level_schema[parent_index + 1]
        return None

    def generate_initial_terms(
        self: "Ontology",
        num_classes: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Generate a structured multi-level taxonomic skeleton from the domain using the LLM."""
        level_descriptions = []
        for index, level in enumerate(self.level_schema):
            level_num = index + 1
            if index == 0:
                level_descriptions.append(
                    f"LEVEL {level_num}: {level.name.upper()} - Top-level abstract categories in {self.domain}"
                )
            elif level.is_lexical:
                parent_level = self.level_schema[index - 1]
                level_descriptions.append(
                    f"LEVEL {level_num}: {level.name.upper()} - Natural-language aliases or surface forms "
                    f"people may use for each {parent_level.name}"
                )
            else:
                parent_level = self.level_schema[index - 1]
                level_descriptions.append(
                    f"LEVEL {level_num}: {level.name.upper()} - Specific items within each {parent_level.name}"
                )

        hierarchy_description = "\n".join(level_descriptions)
        example_schema = self._build_example_schema()
        scope_description = (
            f"SCOPE: {self.scope_description}"
            if self.scope_description else ""
        )

        instructions = (
            "You are an ontology engineer specialist. Generate accurate, specific taxonomies "
            "for the given domain based on the requested hierarchy."
        )

        prompt = f"""You are an ontology engineer specialist. Generate a detailed
            taxonomic hierarchy for the given domain.

            DOMAIN: {self.domain}
            {scope_description}

            TASK:
            Generate a structured taxonomy with {len(self.level_schema)} levels:
            {hierarchy_description}

            {self._build_count_instructions(num_classes)}

            EXPECTED JSON STRUCTURE:
            {example_schema}

            CRITICAL REQUIREMENTS:
            - Return ONLY valid JSON (no markdown code blocks or preamble)
            - Each entry must have a unique name and description
            - All descriptions must be concise (1-2 sentences)
            - Ensure entries are concrete, recognizable examples for the domain "{self.domain}"
            - Do NOT encapsulate the output in code blocks

            START YOUR RESPONSE WITH {{ CHARACTER. OUTPUT ONLY VALID JSON."""

        try:
            raw_response = self.agent.chat(
                instructions=instructions,
                input=prompt,
            )
            logger.debug(
                "Raw seed response (first 500 chars): %s",
                raw_response[:500],
            )
        except Exception as exc:
            logger.error("LLM chat call failed: %s", exc)
            return None

        try:
            seed_dict = json.loads(raw_response)

            if not isinstance(seed_dict, dict):
                logger.error("Seed response is not a dict: %s",
                             type(seed_dict))
                return None

            if "domain" not in seed_dict or "taxonomy" not in seed_dict:
                logger.error(
                    "Seed response missing 'domain' or 'taxonomy' keys")
                return None

            if not isinstance(seed_dict.get("taxonomy"), list):
                logger.error("'taxonomy' value is not a list")
                return None

            if len(seed_dict.get("taxonomy", [])) == 0:
                logger.error("'taxonomy' list is empty")
                return None

            self.seed = seed_dict
            logger.info(
                "Parsed seed successfully: domain=%s, taxonomy count=%d",
                seed_dict["domain"],
                len(seed_dict["taxonomy"]),
            )
            return seed_dict

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse seed JSON: %s. Raw response (first 300 chars): %s",
                exc,
                raw_response[:300],
            )
            return None

    def _build_count_instructions(self: "Ontology", num_classes: int) -> str:
        """Build count instruction lines for the seed prompt, safe for any schema depth."""
        lines = [
            (
                f"Generate exactly {num_classes} top-level "
                f"{self.level_schema[0].pluralized_name.lower()}."
            )
        ]
        counts = [(2, 4), (2, 3), (1, 3)]
        for index in range(len(self.level_schema) - 1):
            parent_name = self.level_schema[index].name.lower()
            child_name = self.level_schema[index + 1].pluralized_name.lower()
            lo, hi = counts[min(index, len(counts) - 1)]
            suffix = " (if applicable)" if index > 0 else ""
            lines.append(
                f"For each {parent_name}, include {lo}-{hi} {child_name}{suffix}."
            )
        return "\n".join(lines)

    def _build_example_schema(self: "Ontology") -> str:
        """Build a JSON schema example string from self.level_schema."""
        result = {
            "domain": self.domain,
            "taxonomy": [],
        }

        root_level = self.level_schema[0]
        current = {
            root_level.seed_key: f"Example {root_level.name.capitalize()}",
            "description": "Brief description",
        }

        innermost = current
        for level_idx in range(1, len(self.level_schema)):
            level = self.level_schema[level_idx]
            parent_level = self.level_schema[level_idx - 1]

            if parent_level.children_key is not None:
                child_item = {
                    level.seed_key: f"Example {level.name.capitalize()}",
                    "description": "Brief description",
                }

                if level.children_key is not None:
                    child_item[level.children_key] = [
                        {"term": "Example instance", "description": "..."},
                        {"term": "Another instance", "description": "..."},
                    ]

                innermost[parent_level.children_key] = [child_item]
                innermost = child_item

        result["taxonomy"].append(current)
        return json.dumps(result, indent=2)

    def create_seed_ontology(self: "Ontology") -> None:
        """Create the initial ontology skeleton from the structured seed."""
        if self.seed is None:
            logger.error("seed is None - call generate_initial_terms() first")
            return

        self.ontology_graph.clear()

        root_level = self.level_schema[0]
        taxonomy = self.seed.get("taxonomy", [])
        for root_item in taxonomy:
            self._process_taxonomy_item(root_item, root_level, parent_id=None)

        logger.info("Created seed ontology with %d nodes",
                    len(self.ontology_graph))

    def _process_taxonomy_item(
        self: "Ontology",
        item: Dict[str, Any],
        level: OntologyLevel,
        parent_id: Optional[str],
    ) -> Optional[str]:
        """Recursively process a single taxonomy item from the seed."""
        term = item.get(level.seed_key)
        if not term:
            logger.warning("Item missing '%s': %s", level.seed_key, item)
            return None

        description = item.get("description", "")

        if term in self.ontology_graph:
            logger.warning(
                "Duplicate term '%s' at level '%s' - skipping",
                term,
                level.name,
            )
            return term

        # Preserve any extra annotations from the seed item (e.g. pbi_field,
        # pbi_table, dax_template) as node attributes.  Only scalar values are
        # kept — nested structures (lists/dicts) belong to children_key or
        # other non-annotation fields.
        reserved_keys = {level.seed_key, "description"}
        if level.children_key:
            reserved_keys.add(level.children_key)
        extra_attrs = {
            k: v
            for k, v in item.items()
            if k not in reserved_keys and isinstance(v, (str, int, float, bool))
        }

        self.ontology_graph.add_node(
            term,
            term=term,
            description=description,
            level=level.name,
            is_lexical=level.is_lexical,
            n_visits=0,
            total_reward=0.0,
            **extra_attrs,
        )

        if parent_id is not None and level.relation_to_parent:
            self.ontology_graph.add_edge(
                parent_id,
                term,
                relation=level.relation_to_parent,
            )

        child_level = self._get_child_level(level.name)
        if child_level and level.children_key:
            children = item.get(level.children_key, [])
            for child_item in children:
                self._process_taxonomy_item(
                    child_item,
                    child_level,
                    parent_id=term,
                )

        return term
