"""Helpers for converting ontology graphs into Dash Cytoscape elements."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import networkx as nx


RELATION_LABELS = {
    "subClassOf": "rdfs:subClassOf",
    "type": "rdf:type",
}

EXPAND_CONTEXT_MENU_ID = "expand-ontology-from-node"
EXPAND_CONTEXT_MENU: List[Dict[str, Any]] = [
    {
        "id": EXPAND_CONTEXT_MENU_ID,
        "label": "Expand ontology from this node",
        "availableOn": ["node"],
    }
]


CYTOSCAPE_STYLESHEET: List[Dict[str, Any]] = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "font-size": 12,
            "font-family": "IBM Plex Sans, Arial, sans-serif",
            "text-wrap": "wrap",
            "text-max-width": 130,
            "text-valign": "center",
            "text-halign": "center",
            "width": 72,
            "height": 72,
            "background-color": "#9ca3af",
            "color": "#111827",
            "border-width": 2,
            "border-color": "#111827",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "arrow-scale": 1,
            "line-color": "#374151",
            "target-arrow-color": "#374151",
            "label": "data(label)",
            "font-size": 10,
            "font-family": "IBM Plex Mono, monospace",
            "text-background-color": "#f8fafc",
            "text-background-opacity": 1,
            "text-background-padding": 2,
            "text-rotation": "autorotate",
            "width": 2,
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "border-width": 4,
            "border-color": "#0f172a",
            "overlay-opacity": 0,
        },
    },
    {
        "selector": ".class",
        "style": {
            "shape": "rectangle",
            "background-color": "#3b82f6",
            "color": "#f8fafc",
        },
    },
    {
        "selector": ".subclass",
        "style": {
            "shape": "roundrectangle",
            "background-color": "#14b8a6",
            "color": "#0f172a",
        },
    },
    {
        "selector": ".instance",
        "style": {
            "shape": "ellipse",
            "background-color": "#f97316",
            "color": "#111827",
        },
    },
    {
        "selector": ".retired",
        "style": {
            "border-style": "dashed",
            "border-color": "#7c2d12",
            "opacity": 0.7,
        },
    },
    {
        "selector": '.relation-type',
        "style": {
            "line-style": "dashed",
        },
    },
]


def relation_to_label(relation: str) -> str:
    """Map an internal relation name to a display label."""
    return RELATION_LABELS.get(relation, relation)


def graph_to_cytoscape(
    graph: nx.DiGraph,
    expandable_nodes: Sequence[str],
) -> List[Dict[str, Any]]:
    """Convert an ontology DiGraph to Dash Cytoscape elements."""
    expandable = set(expandable_nodes)
    elements: List[Dict[str, Any]] = []

    for node_id, attrs in graph.nodes(data=True):
        level = str(attrs.get("level", "unknown"))
        classes = [level]
        if attrs.get("retired", False):
            classes.append("retired")
        elements.append(
            {
                "data": {
                    "id": str(node_id),
                    "label": attrs.get("term", str(node_id)),
                    "description": attrs.get("description", ""),
                    "level": level,
                    "visits": attrs.get("n_visits", 0),
                    "reward": attrs.get("total_reward", 0.0),
                    "expandable": str(node_id) in expandable,
                    "retired": attrs.get("retired", False),
                },
                "classes": " ".join(classes),
            }
        )

    for source, target, attrs in graph.edges(data=True):
        relation = str(attrs.get("relation", ""))
        edge_classes: List[str] = []
        if relation == "type":
            edge_classes.append("relation-type")
        elements.append(
            {
                "data": {
                    "id": f"{source}->{target}",
                    "source": str(source),
                    "target": str(target),
                    "label": relation_to_label(relation),
                    "relation": relation,
                },
                "classes": " ".join(edge_classes),
            }
        )

    return elements


def build_node_details(
    graph: nx.DiGraph,
    node_id: str,
    expandable_nodes: Sequence[str],
) -> Dict[str, Any]:
    """Collect node details for the explorer side panel."""
    if node_id not in graph:
        raise KeyError(f"Node '{node_id}' not found in graph")

    attrs = graph.nodes[node_id]
    parents = [str(parent) for parent in graph.predecessors(node_id)]
    children = [str(child) for child in graph.successors(node_id)]

    incoming_relations = [
        {
            "direction": "from",
            "node": str(parent),
            "relation": relation_to_label(str(graph.edges[parent, node_id].get("relation", ""))),
        }
        for parent in graph.predecessors(node_id)
    ]
    outgoing_relations = [
        {
            "direction": "to",
            "node": str(child),
            "relation": relation_to_label(str(graph.edges[node_id, child].get("relation", ""))),
        }
        for child in graph.successors(node_id)
    ]

    visits = int(attrs.get("n_visits", 0) or 0)
    total_reward = float(attrs.get("total_reward", 0.0) or 0.0)
    mean_reward = total_reward / visits if visits else 0.0

    return {
        "id": str(node_id),
        "term": attrs.get("term", str(node_id)),
        "description": attrs.get("description", ""),
        "level": attrs.get("level", "unknown"),
        "parents": parents,
        "children": children,
        "incoming_relations": incoming_relations,
        "outgoing_relations": outgoing_relations,
        "expandable": str(node_id) in set(expandable_nodes),
        "retired": attrs.get("retired", False),
        "visits": visits,
        "total_reward": total_reward,
        "mean_reward": mean_reward,
    }
