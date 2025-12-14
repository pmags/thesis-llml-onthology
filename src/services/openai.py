## This module includes methods to call and use openai services

import os
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

    # get environment variables
    def _get_env_variables(self) -> None:
        """_summary_
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
