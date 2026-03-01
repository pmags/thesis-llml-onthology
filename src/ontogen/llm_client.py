"""
LLM client wrapper for OpenAI API interactions.

This module provides the ChatGpt class for sending prompts to OpenAI models
and retrieving structured responses for ontology generation tasks.
"""

import os
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI


class ChatGpt:
    """
    A client wrapper for interacting with OpenAI's Chat API.

    Provides a simplified interface for sending chat requests to OpenAI's
    language models. Handles API key configuration through environment
    variables or direct initialization.

    Attributes:
        model: The OpenAI model identifier to use for chat completions.
        api_key: The API key for authenticating with OpenAI services.
        client: The initialized OpenAI client instance.

    Example:
        >>> chat_gpt = ChatGpt(model="gpt-4")
        >>> response = chat_gpt.chat(
        ...     instructions="You are a helpful assistant.",
        ...     input="What is Python?"
        ... )
        >>> print(response)
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or self._get_env_variables()
        self.client = OpenAI(api_key=self.api_key)

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
            model=self.model,
            instructions=instructions,
            input=input,
        )
        return response.output_text

    def get_similarity(self, pairs_table: pd.DataFrame) -> pd.DataFrame:
        """
        Get similarity between pairs of terms using OpenAI API.

        Uses a simple 0-10 scale prompt. This is the legacy method;
        prefer get_similarity_with_descriptions() for new code.

        Args:
            pairs_table: DataFrame with columns 'category_x' and 'category_y'
                representing pairs of terms.

        Returns:
            DataFrame with an additional column 'similarity' indicating the
            similarity score between the term pairs.
        """
        # TODO: This method is dependent on specific column names in the input DataFrame.
        terms_pairs_json = pairs_table.to_json(orient='records')

        prompt_template = f"""
        In this survey you'll be asked to rate quantitatively, on a scale, the intensity of the
        semantic relatedness between pairs of affective words. Please, before starting, read
        carefully the instructions and the examples provided.

        The question we're asking is: how much related are the two words? Vaguely related words
        should be scored with lower values, and strongly related words with higher values. Please
        note that opposite words frequently present high values of relatedness.

        For example, the words 'modest' and 'smart' don't seem very related. 'Conceal' and 'mask'
        seem very related. 'Confident' is highly related with itself. 'Violent' and 'pacific',
        being opposite words, are frequently related, just like 'happiness' and 'sadness'.

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
            instructions=(
                "You are a taxonomy and ontology expert. "
                "Provide concise and accurate responses based on the user's queries."
            ),
            input=prompt_template,
        )

        df = pd.DataFrame(
            json.loads(calc),
            columns=["category_x", "category_y", "similarity"],
        )
        return df

    def get_similarity_with_descriptions(
        self,
        term_x: str,
        description_x: str,
        term_y: str,
        description_y: str,
    ) -> Dict[str, Any]:
        """
        Get similarity between a single pair of concepts using OpenAI API.

        Each concept consists of a term and its description. Returns a 0-100
        integer similarity score determined via prompt-based evaluation.

        Args:
            term_x: First term/concept name.
            description_x: Description of the first concept.
            term_y: Second term/concept name.
            description_y: Description of the second concept.

        Returns:
            Dictionary with keys 'term_x', 'description_x', 'term_y',
            'description_y', 'similarity'.
        """
        prompt_template = f"""
        You are evaluating the semantic relatedness between pairs of concepts. Each concept
        consists of a TERM and its DESCRIPTION within a specific domain.

        TASK: Rate the intensity of the semantic relatedness between concept pairs on a scale
        from 0 to 100, where:
        - 0 = Not related at all
        - 100 = Highly related (including opposites, synonyms, or very closely connected concepts)

        IMPORTANT GUIDELINES:
        1. Consider BOTH the term and its description when evaluating relatedness
        2. Vaguely related concepts should receive lower scores (0-30)
        3. Moderately related concepts should receive middle scores (40-60)
        4. Strongly related concepts should receive higher scores (70-100)
        5. Note that OPPOSITE concepts often have HIGH relatedness scores because they are
           semantically connected (e.g., "hot" and "cold" are highly related as temperature
           opposites)
        6. Use any integer value between 0 and 100, not just multiples of 10

        EXAMPLES WITH EXPLANATIONS:

        Example 1 - Low Relatedness (Score: 10):
        {{
        "term_x": "modest",
        "description_x": "Having or showing a moderate estimation of one's own abilities",
        "term_y": "smart",
        "description_y": "Having or showing intelligence and mental capability",
        "similarity": 10
        }}

        Example 2 - High Relatedness (Score: 90):
        {{
        "term_x": "conceal",
        "description_x": "To hide something or prevent it from being known or seen",
        "term_y": "mask",
        "description_y": "To disguise or cover something to prevent identification",
        "similarity": 90
        }}

        Example 3 - Perfect Relatedness (Score: 100):
        {{
        "term_x": "confident",
        "description_x": "Feeling or showing certainty about something; self-assured",
        "term_y": "confident",
        "description_y": "Feeling or showing certainty about something; self-assured",
        "similarity": 100
        }}

        CONCEPT PAIR TO EVALUATE:
        {{
            "term_x": "{term_x}",
            "description_x": "{description_x}",
            "term_y": "{term_y}",
            "description_y": "{description_y}"
        }}

        RESPONSE FORMAT:
        Return ONLY a JSON object with this exact structure. Do NOT encapsulate in code blocks
        or markdown:
        {{
            "term_x": "{term_x}",
            "description_x": "{description_x}",
            "term_y": "{term_y}",
            "description_y": "{description_y}",
            "similarity": <integer between 0 and 100>
        }}
        """

        response_text = self.chat(
            instructions=(
                "You are a taxonomy and ontology expert. "
                "Provide concise and accurate responses based on the user's queries."
            ),
            input=prompt_template,
        )

        try:
            result = json.loads(response_text)
            if isinstance(result.get('similarity'), float):
                result['similarity'] = int(result['similarity'])

            if not isinstance(result.get('similarity'), int) or not (0 <= result['similarity'] <= 100):
                raise ValueError(f"Invalid similarity score: {result.get('similarity')}")

            return result
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response for pair ({term_x}, {term_y}): {str(e)}")
            return {
                "term_x": term_x,
                "description_x": description_x,
                "term_y": term_y,
                "description_y": description_y,
                "similarity": None,
            }

    def get_similarity_with_descriptions_batch(
        self, pairs: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple (term, description) pairs in a single LLM call.

        Args:
            pairs: List of dicts with keys 'term_x', 'description_x',
                'term_y', 'description_y'.

        Returns:
            List of dicts (same order) with added key 'similarity' as
            integer 0-100.
        """
        if not pairs:
            return []

        pairs_json = json.dumps(pairs, ensure_ascii=False)

        prompt = f"""You are evaluating the semantic relatedness between multiple pairs of concepts.
Each concept consists of a TERM and its DESCRIPTION within a specific domain.

TASK: For each input pair, rate the intensity of semantic relatedness on a scale from 0 to 100
(integer):
- 0 = Not related at all
- 100 = Highly related (including opposites, synonyms, or very closely connected concepts)

IMPORTANT GUIDELINES:
1. Consider BOTH the term and its description when evaluating relatedness.
2. Vaguely related concepts should receive lower scores (0-30).
3. Moderately related concepts should receive middle scores (40-60).
4. Strongly related concepts should receive higher scores (70-100).
5. Opposite concepts can be highly related (e.g., "hot" vs "cold").
6. Use an integer value between 0 and 100.

CONCEPT PAIRS TO EVALUATE (preserve order):
{pairs_json}

RESPONSE FORMAT:
- Return ONLY a JSON ARRAY (no markdown, no code fences)
- Preserve the INPUT ORDER in your output
- Each item MUST contain exactly the keys: 'term_x','description_x','term_y','description_y',
  'similarity'
- 'similarity' must be an integer between 0 and 100
"""

        response_text = self.chat(
            instructions=(
                "You are a taxonomy and ontology expert. "
                "Provide concise JSON-only output in the same format as the inputs."
            ),
            input=prompt,
        )

        try:
            result = json.loads(response_text)
            if isinstance(result, list):
                for item in result:
                    sim = item.get("similarity")
                    if isinstance(sim, float):
                        item["similarity"] = int(sim)
                return result
            raise ValueError("batch response is not a list")
        except Exception as e:
            print(f"Error parsing batch similarity response: {e}")
            return [
                {
                    "term_x": p.get("term_x"),
                    "description_x": p.get("description_x"),
                    "term_y": p.get("term_y"),
                    "description_y": p.get("description_y"),
                    "similarity": None,
                }
                for p in pairs
            ]

    def generate_cluster_name(self, terms: List[str]) -> str:
        """
        Generate a concise descriptive name for a cluster of related terms.

        Args:
            terms: A list of terms that share a common theme or concept.

        Returns:
            A single-word or short-phrase name that best describes the common
            theme among the provided terms.
        """
        prompt_template = f"""
        You are a helpful assistant that generates concise and descriptive names for clusters
        of related terms. Given the following list of terms, provide a single-word or
        short-phrase name that best describes the common theme or concept among them.
        Do not encapsulate the output in code blocks, bold or any markdown.

        Terms: {terms}
        """

        response = self.chat(
            instructions=(
                "You are a taxonomy and ontology expert. "
                "Provide concise and accurate responses based on the user's queries."
            ),
            input=prompt_template,
        )
        return response.strip()

    def _get_env_variables(self) -> Optional[str]:
        """
        Fetch OpenAI API key from environment variables.

        Returns:
            The OpenAI API key string, or None if not found.
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        return self.api_key
