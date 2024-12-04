import openai
from src.llm_client.base import LLMClient

class OpenAIClient(LLMClient):
    """
    OpenAI API 的具體實作客戶端。
    """

    def __init__(self, api_key: str):
        """
        Initialize the OpenAI client with the given API key.

        :param api_key: The API key for OpenAI.
        :type api_key: str
        """
        super().__init__(api_key)
        openai.api_key = self.api_key

    def generate_text(self, prompt: str, model: str = "gpt-3.5-turbo", **kwargs) -> dict:
        self.validate_key()
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response
        except openai.error.AuthenticationError:
            raise RuntimeError("Authentication failed. Please check your API key.")
        except openai.error.RateLimitError:
            raise RuntimeError("Rate limit exceeded. Please try again later.")
        except openai.error.OpenAIError as e:
            raise RuntimeError(f"OpenAI API Error: {e}")

    def validate_key(self):
        """
        驗證 API 金鑰是否設置。
        """
        if not self.api_key:
            raise ValueError("API key is not provided.")
        
    def extract_keywords(self, content: str, prompt: str) -> dict:
        """
        Extract keywords from the given content using the OpenAI API.

        :param content: The content to extract keywords from.
        :type content: str
        :param prompt: The prompt to use for keyword extraction.
        :type prompt: str
        :return: The extracted keywords.
        :rtype: dict
        """
        response = self.generate_text(content, prompt)
        return response.choices[0].message.content
    
    def generate_summary(self, content: str, prompt: str) -> dict:

        response = self.generate_text(content, prompt)
        return response.choices[0].message.content
    
    def generate_news_summary(self, content: str, prompt: str) -> dict:
        """
        Generate a summary of the news article using the OpenAI API.

        :param content: The content of the news article.
        :type content: str
        :param prompt: The prompt to use for generating the summary.
        :type prompt: str
        :return: The generated summary.
        :rtype: dict
        """
        response = self.generate_text(content, prompt)
        return response.choices[0].message.content