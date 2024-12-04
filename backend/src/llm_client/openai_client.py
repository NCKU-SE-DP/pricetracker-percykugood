import openai
from typing import Dict, Optional
from .base import LLMClient, MessageInterface


class OpenAIClient(LLMClient):
    """
    OpenAI API 的具體實作客戶端。
    """

    def __init__(self, api_key: str):
        """
        初始化 OpenAI 客戶端。

        :param api_key: OpenAI 的 API 金鑰。
        :type api_key: str
        """
        super().__init__(api_key)
        openai.api_key = self.api_key

    def generate_text(self, prompt: MessageInterface, model: str = "gpt-3.5-turbo", **kwargs) -> Optional[Dict]:
        """
        使用 OpenAI API 生成文字內容。

        :param prompt: 提示文字。
        :type prompt: MessageInterface
        :param model: 使用的模型名稱，默認為 "gpt-3.5-turbo"。
        :type model: str
        :param kwargs: 其他選項，例如 max_tokens、temperature。
        :return: OpenAI 的回應內容。
        :rtype: str
        """
        ai_completion = openai.ChatCompletion.create(
            model=model,
            messages=prompt.generate_prompt,
            **kwargs
        )
        return ai_completion.choices[0].message.content



    def validate_key(self):
        """
        驗證 API 金鑰是否設置。
        """
        if not self.api_key:
            raise ValueError("API key is not provided.")

    def extract_keywords(self, content: str, prompt_template: str) -> str:
        """
        使用 OpenAI API 提取關鍵字。

        :param content: 需要提取關鍵字的內容。
        :type content: str
        :param prompt_template: 關鍵字提取的提示模板。
        :type prompt_template: str
        :return: 提取的關鍵字。
        :rtype: str
        """
        prompt = prompt_template.format(content=content)
        response = self.generate_text(prompt)
        if not response:
            raise RuntimeError("Failed to extract keywords: OpenAI API returned None")
        return response["choices"][0]["message"]["content"]

    def generate_summary(self, content: str, prompt_template: str) -> str:
        """
        使用 OpenAI API 生成摘要。

        :param content: 需要生成摘要的內容。
        :type content: str
        :param prompt_template: 摘要生成的提示模板。
        :type prompt_template: str
        :return: 生成的摘要。
        :rtype: str
        """
        prompt = prompt_template.format(content=content)
        response = self.generate_text(prompt)
        if not response:
            raise RuntimeError("Failed to generate summary: OpenAI API returned None")
        return response["choices"][0]["message"]["content"]
