from abc import ABC, abstractmethod
from .exception import DomainMismatchException
from pydantic import BaseModel, Field

class MessagePassingInterfaceExample(BaseModel):
    key: str = Field(
        default=...,
        example="example",
        description="description"
    )

class MessageInterface(BaseModel):
    system_content: str = Field(...)
    user_content: str = Field(...)
    
    @property
    def generate_prompt(self):
        return [{"role": "system", "content": f"{self.system_content}"},
                {"role": "user", "content": f"{self.user_content}"}]

class LLMClient(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def generate_text(self, prompt: MessageInterface) -> dict:
        """
        抽象方法，用於生成文字內容。
        具體實作需要在子類別中完成。
        :param prompt: 提示文字
        :param kwargs: 其他可選參數（如溫度、最大 token 數）
        :return: 回應內容（字典格式）
        """
        pass

    def validate_key(self):
        """
        驗證 API 金鑰是否設置，子類別可以覆蓋此方法以執行額外檢查。
        """
        if not self.api_key:
            raise ValueError("API key is not provided.")

    @abstractmethod
    def extract_keywords(self, content: str, prompt: MessageInterface) -> dict:
        """
        抽象方法，用於從給定內容中提取關鍵字。
        具體實作需要在子類別中完成。
        :param content: 要提取關鍵字的內容
        :param prompt: 用於關鍵字提取的提示
        :return: 提取的關鍵字
        """
        pass

    @abstractmethod
    def generate_summary(self, content: str, prompt: MessageInterface) -> dict:
        """
        抽象方法，用於生成摘要。
        具體實作需要在子類別中完成。
        :param content: 要生成摘要的內容
        :param prompt: 用於生成摘要的提示
        :return: 生成的摘要
        """
        pass