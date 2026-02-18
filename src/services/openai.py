## This module includes methods to call and use openai services

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


class ChatGpt:
    """
    A client wrapper for interacting with OpenAI's Chat API.
    This class provides a simplified interface for sending chat requests to OpenAI's
    language models. It handles API key configuration through environment variables
    or direct initialization.
    Attributes:
        model (str): The OpenAI model identifier to use for chat completions.
        api_key (str): The API key for authenticating with OpenAI services.
        client (OpenAI): The initialized OpenAI client instance.
    Example:
        >>> chat_gpt = ChatGpt(model="gpt-4")
        >>> response = chat_gpt.chat(
        ...     instructions="You are a helpful assistant.",
        ...     input="What is Python?"
        ... )
        >>> print(response)
    """  

    def __init__(self, model:str = None, api_key: str = None) -> None:

        self.model = model
        self.api_key = api_key or self._get_env_variables()
        self.client = OpenAI(
            api_key= self.api_key,
        )

    def chat(self, instructions: str, input: str) -> str:
        """
        Send a chat request to OpenAI API.
        
        Args:
            instructions: System instructions for the chat model.
            input: User input message to send to the model.
            
        Returns:
            The output text from the model's response.
        """
        response = self.client.responses.create(
            model = self.model,
            instructions = instructions,
            input = input
        )
        return response.output_text

    def get_similarity(self, pairs_table: pd.DataFrame) -> pd.DataFrame:
        """_summary_
        Function to get similarity between pairs of terms using OpenAI API

        Args:
            pairs_table (pd.DataFrame): DataFrame with columns 'category_x' and 'category_y' representing pairs of terms.

        Returns:
            pd.DataFrame: DataFrame with an additional column 'similarity' indicating the similarity score between the term pairs.
        """        
        
        # TODO: This method is dependent on specific column names in the input DataFrame.
        
        # serialize into a json to make it simpler to add to prompt
        
        terms_pairs_json = pairs_table.to_json(orient='records')
        
        prompt_template =f"""
        In this survey you'll be asked to rate quantitatively, on a scale, the intensity of the semantic relatedness between pairs 
        of affective words. Please, before starting, read carefully the instructions and the examples provided.

        The question we're asking is: how much related are the two words? Vaguely related words should be scored with lower 
        values, and strongly related words with higher values. Please note that opposite words frequently present high values 
        of relatedness.

        For example, the words 'modest' and 'smart' don't seem very related. 'Conceal' and 'mask' seem very related. 
        'Confident' is highly related with itself. 'Violent' and 'pacific', being opposite words, are frequently related, 
        just like 'happiness' and 'sadness'.

        Examples:
        [
            {{"category_x":"modest","category_y":"smart","similarity":1}},
            {{"category_x":"conceal","category_y":"mask","similarity":7}},
            {{"category_x":"confident","category_y":"confident","similarity":10}},
            {{"category_x":"violent","category_y":"pacific","similarity":8}},
            {{"category_x":"happiness","category_y":"sadness","similarity":10}}
        ]

        Using the schema i provide and not encapsulate the output in code blocks, 
        please rate from 0 to 10 the semantic relatedness of the following pairs of words, 
        with 0 indicating words not related at all and 10 indicating very related words:
        {terms_pairs_json}
        """
        
        calc = self.chat(
            instructions="You are a taxonomy and ontology expert. Provide concise and accurate responses based on the user's queries.",
            input=prompt_template
        )
        
        df = pd.DataFrame(json.loads(calc), columns=["category_x", "category_y", "similarity"])
        
        return df

    def get_similarity_with_descriptions(
        self, 
        term_x: str, 
        description_x: str, 
        term_y: str, 
        description_y: str
    ) -> dict:
        """
            Function to get similarity between a single pair of concepts using OpenAI API.
            Each concept consists of a term and its description.

            Args:
                term_x (str): First term/concept name.
                description_x (str): Description of the first concept.
                term_y (str): Second term/concept name.
                description_y (str): Description of the second concept.

            Returns:
                dict: Dictionary with keys 'term_x', 'description_x', 'term_y', 'description_y', 'similarity'
        """        
        
        
        prompt_template =f"""
        You are evaluating the semantic relatedness between pairs of concepts. Each concept consists of a TERM and its DESCRIPTION within a specific domain.
        
        TASK: Rate the intensity of the semantic relatedness between concept pairs on a scale from 0 to 100, where:
        - 0 = Not related at all
        - 100 = Highly related (including opposites, synonyms, or very closely connected concepts)
        
        IMPORTANT GUIDELINES:
        1. Consider BOTH the term and its description when evaluating relatedness
        2. Vaguely related concepts should receive lower scores (0-30)
        3. Moderately related concepts should receive middle scores (40-60)
        4. Strongly related concepts should receive higher scores (70-100)
        5. Note that OPPOSITE concepts often have HIGH relatedness scores because they are semantically connected (e.g., "hot" and "cold" are highly related as temperature opposites)
        6. Use any integer value between 0 and 100, not just multiples of 10. For example, 25, 45, 67, 85 are all valid scores.
        
        EXAMPLES WITH EXPLANATIONS:

        Example 1 - Low Relatedness (Score: 10):
        {{
        "term_x": "modest",
        "description_x": "Having or showing a moderate estimation of one's own abilities; humble and unassuming",
        "term_y": "smart",
        "description_y": "Having or showing intelligence and mental capability",
        "similarity": 10
        }}
        Reasoning: These personality traits don't share a strong semantic connection.

        Example 2 - High Relatedness (Score: 90):
        {{
        "term_x": "conceal",
        "description_x": "To hide something or prevent it from being known or seen",
        "term_y": "mask",
        "description_y": "To disguise or cover something to prevent identification",
        "similarity": 90
        }}
        Reasoning: Both concepts involve hiding or obscuring, making them near-synonyms.

        Example 3 - Perfect Relatedness (Score: 100):
        {{
        "term_x": "confident",
        "description_x": "Feeling or showing certainty about something; self-assured",
        "term_y": "confident",
        "description_y": "Feeling or showing certainty about something; self-assured",
        "similarity": 100
        }}
        Reasoning: Identical concepts are perfectly related.

        Example 4 - Opposite but Highly Related (Score: 90):
        {{
        "term_x": "violent",
        "description_x": "Using or involving physical force intended to hurt, damage, or kill",
        "term_y": "pacific",
        "description_y": "Peaceful in character or intent; promoting peace and calm",
        "similarity": 90
        }}
        Reasoning: These are opposites on the aggression spectrum, but highly semantically connected.

        Example 5 - Opposite Emotional States (Score: 100):
        {{
        "term_x": "happiness",
        "description_x": "A state of emotional well-being characterized by positive feelings",
        "term_y": "sadness",
        "description_y": "A state of emotional distress characterized by negative feelings and sorrow",
        "similarity": 100
        }}
        Reasoning: These are fundamental opposite emotions, making them maximally related on the emotional spectrum.

        CONCEPT PAIR TO EVALUATE:
        {{
            "term_x": "{term_x}",
            "description_x": "{description_x}",
            "term_y": "{term_y}",
            "description_y": "{description_y}"
        }}

        RESPONSE FORMAT:
        Return ONLY a JSON object with this exact structure. Do NOT encapsulate in code blocks or markdown:
        {{
            "term_x": "{term_x}",
            "description_x": "{description_x}",
            "term_y": "{term_y}",
            "description_y": "{description_y}",
            "similarity": <integer between 0 and 100>
        }}
        """
        
        response_text = self.chat(
            instructions="You are a taxonomy and ontology expert. Provide concise and accurate responses based on the user's queries.",
            input=prompt_template
        )
        
        try:
            result = json.loads(response_text)
            if isinstance(result.get('similarity'), float):
                result['similarity'] = int(result['similarity'])
            
            if not isinstance(result.get('similarity'), int) or not (0 <= result['similarity'] <= 100):
                raise ValueError(f"Invalid similarity score: {result.get('similarity')}")
            
            return result
        except (json.JSONDecodeError, ValueError) as e:
            # Return error result with None similarity
            print(f"Error parsing response for pair ({term_x}, {term_y}): {str(e)}")
            return {
                "term_x": term_x,
                "description_x": description_x,
                "term_y": term_y,
                "description_y": description_y,
                "similarity": None
            }

    def generate_cluster_name(self, terms: list[str]) -> str:
        """
        Generates a concise and descriptive name for a cluster of related terms.
        Args:
            terms (list of str): A list of terms that share a common theme or concept.
        Returns:
            str: A single-word or short-phrase name that best describes the common theme or concept among the provided terms.
        """
        
        
        prompt_template =f"""
        You are a helpful assistant that generates concise and descriptive names for clusters of related terms.
        Given the following list of terms, provide a single-word or short-phrase name that best describes the common theme or concept among them.
        Do not encapsulate the output in code blocks, bold or any markdown.
        
        Terms: {terms}
        """
    
        response = self.chat(
            instructions="You are a taxonomy and ontology expert. Provide concise and accurate responses based on the user's queries.",
            input=prompt_template
        )
        
        return response.strip()

    # get environment variables
    def _get_env_variables(self) -> None:
        """_summary_
            fetch OpenAI API key from environment variables.
        Returns:
            str: The OpenAI API key.
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
