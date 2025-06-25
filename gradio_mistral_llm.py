# import os
# from gradio_client import Client
# from langchain_core.language_models.llms import LLM
# from pydantic import Extra
# from dotenv import load_dotenv

# class GradioMistralLLM(LLM):
#     url: str

#     class Config:
#         extra = Extra.allow  # allow custom fields

#     def __init__(self, url: str):
#         super().__init__(url=url)
#         self._client = Client(url)

#     def _call(self, prompt: str, **kwargs) -> str:
#         result = self._client.predict(
#             message=prompt,
#             param_2=1024,   # max tokens
#             param_3=0.6,    # temperature
#             param_4=0.9,    # top_p
#             param_5=50,     # top_k
#             param_6=1.2,    # repetition penalty
#             api_name="/chat"
#         )
#         return result

#     @property
#     def _llm_type(self) -> str:
#         return "gradio-mistral"

from langchain.llms.base import LLM
from typing import Optional, List
from gradio_client import Client
import streamlit as st
from pydantic import Extra

class GradioMistralLLM(LLM):
    space_id: str

    class Config:
        extra = Extra.allow  # allow custom fields

    def __init__(self, space_id: str):
        super().__init__(space_id=space_id)
        hf_token = st.secrets["HF_TOKEN"]  # Hugging Face token
        self._client = Client(space_id, hf_token=hf_token)

    def _call(self, prompt: str, **kwargs) -> str:
        result = self._client.predict(
            prompt,         # assuming the Space expects one input
            1024,           # max tokens
            0.6,            # temperature
            0.9,            # top_p
            50,             # top_k
            1.2,            # repetition_penalty
            api_name="/chat"
        )
        return result

    @property
    def _llm_type(self) -> str:
        return "gradio-mistral"
