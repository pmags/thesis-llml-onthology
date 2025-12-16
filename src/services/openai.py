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
            {{"category_x":"conceal","category_y":"mask","similarity":4}},
            {{"category_x":"confident","category_y":"confident","similarity":5}},
            {{"category_x":"violent","category_y":"pacific","similarity":4}},
            {{"category_x":"happiness","category_y":"sadness","similarity":5}}
        ]

        Using the schema i provide and not encapsulate the output in code blocks, 
        please rate from 0 to 4 the semantic relatedness of the following pairs of words, 
        with 0 indicating words not related at all and 4 indicating very related words:
        {terms_pairs_json}
        """
        
        calc = self.chat(
            instructions="You are a taxonomy and ontology expert. Provide concise and accurate responses based on the user's queries.",
            input=prompt_template
        )
        
        df = pd.DataFrame(json.loads(calc), columns=["category_x", "category_y", "similarity"])
        
        return df

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
