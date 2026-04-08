"""Visualization helpers for ontology debugging and reporting."""

import io
import logging
import os
from typing import TYPE_CHECKING, Dict

import networkx as nx
import pydotplus
from IPython.display import Image, display
from rdflib import Literal
from rdflib.namespace import RDF, RDFS
from rdflib.tools.rdf2dot import rdf2dot

if TYPE_CHECKING:
    from ontogen.ontology import Ontology


logger = logging.getLogger("ontogen.ontology")


class VisualizationMixin:
    """Mixin containing notebook and HTML visualization helpers."""

    def visualize(self: "Ontology") -> None:
        """Visualize the RDF ontology graph as a PNG image."""
        stream = io.StringIO()
        rdf2dot(self.rdf, stream, opts={})
        graph = pydotplus.graph_from_dot_data(stream.getvalue())
        png_data = graph.create_png()
        display(Image(png_data))

    def visualize_graph(self: "Ontology") -> None:
        """Visualize the internal ontology graph with nodes colored by level."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        color_map = {
            "class": "#4A90D9",
            "subclass": "#50C878",
            "instance": "#FF8C42",
        }

        node_colors = [
            color_map.get(
                self.ontology_graph.nodes[node].get("level", "instance"),
                "#FF8C42",
            )
            for node in self.ontology_graph.nodes
        ]

        _, axis = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(self.ontology_graph, k=2,
                               seed=42, iterations=50)

        nx.draw_networkx_nodes(
            self.ontology_graph,
            pos,
            node_color=node_colors,
            node_size=2000,
            ax=axis,
        )
        nx.draw_networkx_labels(
            self.ontology_graph,
            pos,
            labels={node: node for node in self.ontology_graph.nodes},
            font_size=9,
            font_color="black",
            ax=axis,
        )
        nx.draw_networkx_edges(
            self.ontology_graph,
            pos,
            edge_color="gray",
            arrows=True,
            arrowsize=15,
            ax=axis,
        )

        edge_labels = {
            (source, target): data.get("relation", "")
            for source, target, data in self.ontology_graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.ontology_graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            ax=axis,
        )

        legend_patches = [
            mpatches.Patch(color="#4A90D9", label="Class"),
            mpatches.Patch(color="#50C878", label="Subclass"),
            mpatches.Patch(color="#FF8C42", label="Instance"),
        ]
        axis.legend(handles=legend_patches, loc="upper left", fontsize=10)
        axis.set_title(
            f"Ontology Graph: {self.domain}",
            fontsize=14,
            fontweight="bold",
        )
        axis.axis("off")

        plt.tight_layout()
        plt.show()

    def plot_convergence(self: "Ontology") -> None:
        """Plot convergence diagnostics from the recorded generation history."""
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        if self.history is None or not self.history.expansion_records:
            raise RuntimeError(
                "No generation history available. Run generate_ontology() first."
            )

        dataframe = self.history.to_dataframe()
        plot_df = dataframe[dataframe["node_expanded"].notna()].copy()
        if plot_df.empty:
            print("No expansion iterations to plot.")
            return

        iterations = plot_df["iteration"]
        window = min(3, len(plot_df))
        plot_df["reward_ma"] = plot_df["reward"].rolling(
            window=window,
            min_periods=1,
        ).mean()

        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(
            f"Ontology Generation Convergence — {self.domain}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        ax1 = axes[0, 0]
        ax1.plot(
            iterations,
            plot_df["reward"],
            "o-",
            color="#4A90D9",
            alpha=0.5,
            markersize=5,
            label="Per-iteration reward",
        )
        ax1.plot(
            iterations,
            plot_df["reward_ma"],
            "-",
            color="#2C5F8A",
            linewidth=2.5,
            label=f"Moving avg (w={window})",
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Reward (quality-weighted yield)")
        ax1.set_title("Reward per Iteration")
        ax1.legend(loc="lower right", fontsize=9)
        ax1.set_ylim(bottom=0)
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.bar(
            iterations,
            plot_df["acceptance_rate"],
            color="#50C878",
            alpha=0.7,
            edgecolor="#3A9D5C",
            label="Acceptance rate",
        )
        acceptance_average = plot_df["acceptance_rate"].rolling(
            window=window,
            min_periods=1,
        ).mean()
        ax2.plot(
            iterations,
            acceptance_average,
            "-",
            color="#2D7A47",
            linewidth=2,
            label=f"Moving avg (w={window})",
        )
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Acceptance Rate")
        ax2.set_title("Candidate Acceptance Rate")
        ax2.set_ylim(0, 1.05)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax2.legend(loc="lower right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(
            iterations,
            plot_df["cumulative_nodes"],
            "s-",
            color="#FF8C42",
            linewidth=2,
            markersize=5,
            label="Nodes",
        )
        ax3.plot(
            iterations,
            plot_df["cumulative_edges"],
            "^-",
            color="#D9534F",
            linewidth=2,
            markersize=5,
            label="Edges",
        )
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Count")
        ax3.set_title("Cumulative Graph Growth")
        ax3.legend(loc="upper left", fontsize=9)
        ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        colors = [
            "#D9534F" if plateau >= 3 else "#FFB347" if plateau > 0 else "#50C878"
            for plateau in plot_df["plateau_count"]
        ]
        ax4.bar(
            iterations,
            plot_df["plateau_count"],
            color=colors,
            alpha=0.8,
            edgecolor="gray",
            linewidth=0.5,
        )
        ax4.axhline(
            y=3,
            color="#D9534F",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Early-stop threshold",
        )
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Consecutive Plateaus")
        ax4.set_title("Plateau Counter (convergence signal)")
        ax4.legend(loc="upper left", fontsize=9)
        ax4.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax4.grid(True, alpha=0.3)

        ax5 = axes[2, 0]
        expandable_level_names = frozenset(
            level.name for level in self.level_schema if level.expandable
        )
        visit_counts = [
            self.ontology_graph.nodes[node].get("n_visits", 0)
            for node in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[node].get("level") in expandable_level_names
        ]
        if visit_counts:
            max_visits = max(visit_counts)
            bins = range(0, max_visits + 2)
            ax5.hist(
                visit_counts,
                bins=bins,
                color="#7B68EE",
                alpha=0.8,
                edgecolor="#5B4ACE",
                align="left",
            )
        ax5.set_xlabel("Number of Visits")
        ax5.set_ylabel("Number of Nodes")
        ax5.set_title("Visit Distribution (expandable nodes)")
        ax5.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax5.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax5.grid(True, alpha=0.3)

        ax6 = axes[2, 1]
        node_visits = [
            (node, self.ontology_graph.nodes[node].get("n_visits", 0))
            for node in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[node].get("n_visits", 0) > 0
        ]
        node_visits.sort(key=lambda item: item[1], reverse=True)
        top_10 = node_visits[:10]
        if top_10:
            names = [item[0][:20] for item in reversed(top_10)]
            visits = [item[1] for item in reversed(top_10)]
            level_colors = {self.level_schema[0].name: "#4A90D9"}
            if len(self.level_schema) > 1:
                level_colors[self.level_schema[1].name] = "#50C878"
            if len(self.level_schema) > 2:
                level_colors[self.level_schema[2].name] = "#FF8C42"

            bar_colors = [
                level_colors.get(
                    self.ontology_graph.nodes[item[0]].get("level", ""),
                    "#999999",
                )
                for item in reversed(top_10)
            ]
            bars = ax6.barh(
                names,
                visits,
                color=bar_colors,
                alpha=0.85,
                edgecolor="gray",
                linewidth=0.5,
            )
            for bar, visit_count in zip(bars, visits):
                ax6.text(
                    bar.get_width() + 0.2,
                    bar.get_y() + bar.get_height() / 2,
                    str(visit_count),
                    va="center",
                    fontsize=9,
                )

            from matplotlib.patches import Patch

            legend_handles = [
                Patch(facecolor=color, label=level_name)
                for level_name, color in level_colors.items()
            ]
            ax6.legend(handles=legend_handles, loc="lower right", fontsize=9)
        else:
            ax6.text(
                0.5,
                0.5,
                "No visited nodes",
                ha="center",
                va="center",
                transform=ax6.transAxes,
                fontsize=12,
                color="gray",
            )
        ax6.set_xlabel("Number of Visits")
        ax6.set_title("Top-10 Most Visited Nodes")
        ax6.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax6.grid(True, alpha=0.3, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def visualize_interactive(
        self: "Ontology",
        output_path: str = "ontology.html",
    ) -> str:
        """Render the RDF ontology as a full-page interactive HTML graph using pyvis."""
        try:
            from pyvis.network import Network
        except ImportError as exc:
            raise ImportError(
                "pyvis is required for interactive visualization. "
                "Install it with: pip install pyvis"
            ) from exc

        if self.rdf is None or len(self.rdf) == 0:
            raise RuntimeError(
                "RDF graph is empty. Run build_ontology() first.")

        net = Network(
            notebook=True,
            cdn_resources="in_line",
            height="100vh",
            width="100%",
            directed=True,
        )

        domain_root_color = "#E91E63"
        class_color = "#4A90D9"
        instance_color = "#FF8C42"
        edge_subclass = "#2196F3"
        edge_type = "#4CAF50"

        class_resources = {
            str(subject)
            for subject, _predicate, _object in self.rdf.triples((None, RDF.type, RDFS.Class))
        }

        labels: Dict[str, str] = {}
        for subject, _predicate, obj in self.rdf.triples((None, RDFS.label, None)):
            labels[str(subject)] = str(obj)

        level_map: Dict[str, str] = {}
        desc_map: Dict[str, str] = {}
        visit_map: Dict[str, int] = {}
        reward_map: Dict[str, float] = {}
        for node_id, data in self.ontology_graph.nodes(data=True):
            uri = str(self._sanitize_uri(data.get("term", node_id)))
            level_map[uri] = data.get("level", "")
            desc_map[uri] = data.get("description", "")
            visit_map[uri] = data.get("n_visits", 0)
            reward_map[uri] = data.get("total_reward", 0.0)

        subclass_subjects = {
            str(subject)
            for subject, _predicate, _object in self.rdf.triples((None, RDFS.subClassOf, None))
        }
        top_level_classes = class_resources - subclass_subjects

        domain_uri = str(self._sanitize_uri(self.domain))
        namespace_prefix = str(self.base_namespace)

        def short_uri(uri_str: str) -> str:
            """Shorten ontology URIs for labels and edge captions."""
            if uri_str.startswith(namespace_prefix):
                return uri_str[len(namespace_prefix):]
            return uri_str

        all_nodes: Dict[str, Dict[str, object]] = {
            domain_uri: {
                "label": self.domain,
                "color": domain_root_color,
                "size": 40,
                "title": (
                    f"<b>{self.domain}</b><br>"
                    f"URI: {domain_uri}<br>"
                    "Domain root class"
                ),
                "shape": "dot",
            }
        }

        def ensure_node(uri_str: str) -> None:
            """Register a node with appearance and tooltip metadata."""
            if uri_str in all_nodes:
                return

            label = labels.get(uri_str, short_uri(uri_str))
            level = level_map.get(uri_str, "")
            description = desc_map.get(uri_str, "")
            visits = visit_map.get(uri_str, 0)
            reward = reward_map.get(uri_str, 0.0)
            is_class = uri_str in class_resources

            if uri_str == domain_uri:
                color = domain_root_color
                size = 35
            elif is_class:
                color = class_color
                size = 22
            else:
                color = instance_color
                size = 16

            title_parts = [
                f"<b>{label}</b>",
                f"URI: {uri_str}",
                "rdf:type: rdfs:Class" if is_class else "rdf:type: instance",
                f"Level: {level}" if level else None,
                f"Description: {description}" if description else None,
                f"UCB1 visits: {visits}, reward: {reward:.2f}" if visits else None,
            ]

            all_nodes[uri_str] = {
                "label": label,
                "color": color,
                "size": size,
                "title": "<br>".join(part for part in title_parts if part),
                "shape": "dot",
            }

        edges = []
        rdfs_class_uri = str(RDFS.Class)
        for subject, predicate, obj in self.rdf:
            if isinstance(obj, Literal):
                continue
            if str(subject) == rdfs_class_uri or str(obj) == rdfs_class_uri:
                continue

            subject_str = str(subject)
            object_str = str(obj)

            if predicate == RDFS.subClassOf:
                predicate_label = "rdfs:subClassOf"
                edge_color = edge_subclass
            elif predicate == RDF.type:
                predicate_label = "rdf:type"
                edge_color = edge_type
            else:
                predicate_label = short_uri(str(predicate))
                edge_color = "#999999"

            ensure_node(subject_str)
            ensure_node(object_str)
            edges.append((subject_str, object_str,
                         predicate_label, edge_color))

        for class_uri in top_level_classes:
            ensure_node(class_uri)
            edges.append(
                (class_uri, domain_uri, "rdfs:subClassOf", edge_subclass))

        for node_id, props in all_nodes.items():
            net.add_node(
                node_id,
                label=props["label"],
                title=props["title"],
                color=props["color"],
                size=props["size"],
                shape=props["shape"],
            )

        for source, target, predicate_label, edge_color in edges:
            net.add_edge(
                source,
                target,
                label=predicate_label,
                title=predicate_label,
                arrows="to",
                color=edge_color,
            )

        net.repulsion(
            node_distance=250,
            central_gravity=0.15,
            spring_length=180,
            spring_strength=0.04,
            damping=0.09,
        )

        net.show(output_path)
        abs_path = os.path.abspath(output_path)
        logger.info("Interactive RDF ontology graph saved to %s", abs_path)
        return abs_path
