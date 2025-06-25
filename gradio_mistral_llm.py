from gradio_client import Client
from langchain_core.language_models.llms import LLM
from pydantic import Extra


class GradioMistralLLM(LLM):
    url: str

    class Config:
        extra = Extra.allow  # allow custom fields

    def __init__(self, url: str):
        super().__init__(url=url)
        self._client = Client(url)

    def _call(self, prompt: str, **kwargs) -> str:
        result = self._client.predict(
            message=prompt,
            param_2=1024,   # max tokens
            param_3=0.6,    # temperature
            param_4=0.9,    # top_p
            param_5=50,     # top_k
            param_6=1.2,    # repetition penalty
            api_name="/chat"
        )
        return result

    @property
    def _llm_type(self) -> str:
        return "gradio-mistral"
