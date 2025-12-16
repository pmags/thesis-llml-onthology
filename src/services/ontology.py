from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import DC, RDF, FOAF, RDFS, XSD
import io
import pydotplus
from IPython.display import display, Image
from rdflib.tools.rdf2dot import rdf2dot

class Ontology:
    
    def __init__(self):
        self.terms = None
        self.clusters_df = None
        self.rdf = Graph()
        self.turtle = None
        self.related_namespace = Namespace("http://example.org/ontology/related/")
        self.base_namespace = Namespace("http://example.org/ontology/")

    def build_ontology(self, clusters_df) -> Graph:
        
        """Builds an ontology graph from the given terms and clusters dataframe.

        Args:
            terms (list[str]): A list of terms to include in the ontology.
            clusters_df (DataFrame): A dataframe containing cluster information with columns 'cluster', 'cluster_name', and 'term'.

        Returns:
            Graph: An RDFLib Graph representing the ontology.
        """
        
        g = self.rdf
        g.bind("ex", self.base_namespace)
        g.bind("related", self.related_namespace)
        
        for cluster_id in clusters_df['cluster'].unique():
            cluster_terms = clusters_df[clusters_df['cluster'] == cluster_id]
            cluster_name = cluster_terms.iloc[0]['cluster_name']
        
            # Create URI for the cluster
            cluster_uri = self.base_namespace[cluster_name.replace(" ", "_")]
        
            # Add each term in the cluster
            for _, row in cluster_terms.iterrows():
                term = row['term']
                term_uri = self.base_namespace[term.replace(" ", "_")]
                
                # Term is subclass of cluster
                g.add((term_uri, RDFS.subClassOf, cluster_uri))
                g.add((term_uri, RDFS.label, Literal(term)))
                
                # Add related relationships between terms in the same cluster
                for _, other_row in cluster_terms.iterrows():
                    if row['term'] != other_row['term']:
                        other_term_uri = self.base_namespace[other_row['term'].replace(" ", "_")]
                        g.add((term_uri, self.related_namespace['to'], other_term_uri))
    
        self.rdf = g
        return self.rdf
    
    
    def serialize_ontology(self, format: str = "turtle") -> str:
        """
        Serializes the ontology graph into the specified format.
        
        Args:
            format (str): The serialization format (default is "turtle").
        
        Returns:
            str: The serialized ontology as a string.
        """
        self.turtle = self.rdf.serialize(format=format)
        return self.turtle
    
    
    def visualize(self) -> None:
        """
        Visualizes the ontology graph using pydotplus and IPython display.
        
        Args:
            graph (Graph): The RDFLib graph to visualize.
        """
        stream = io.StringIO()
        rdf2dot(self.rdf, stream, opts = {display})
        graph = pydotplus.graph_from_dot_data(stream.getvalue())
        png_data = graph.create_png()
        display(Image(png_data))