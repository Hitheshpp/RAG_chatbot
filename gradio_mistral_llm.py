# from langchain.llms.base import LLM
# from typing import Optional, List
# from gradio_client import Client
# import streamlit as st
# from pydantic import Extra

# class GradioMistralLLM(LLM):
#     space_id: str

#     class Config:
#         extra = Extra.allow  # allow custom fields

#     def __init__(self, space_id: str):
#         super().__init__(space_id=space_id)
#         hf_token = st.secrets["HF_TOKEN"]  # Hugging Face token
#         self._client = Client(space_id, hf_token=hf_token)

#     def _call(self, prompt: str, **kwargs) -> str:
#         result = self._client.predict(
#             prompt,         # assuming the Space expects one input
#             1024,           # max tokens
#             0.6,            # temperature
#             0.9,            # top_p
#             50,             # top_k
#             1.2,            # repetition_penalty
#             api_name="/chat"
#         )
#         return result

#     @property
#     def _llm_type(self) -> str:
#         return "gradio-mistral"

#------------------------------------------------------------------------------

from langchain.llms.base import LLM
from typing import Optional
from gradio_client import Client
import streamlit as st
from pydantic import Field, model_validator
from pydantic.v1 import Extra  # Required for `Config` in LangChain <0.1.20

class GradioMistralLLM(LLM):
    space_id: str = Field(...)  # Ensure this is declared as a required field
    _client: Optional[Client] = None  # internal use

    class Config:
        extra = Extra.allow  # For LangChain compatibility

    def __init__(self, **data):
        super().__init__(**data)
        hf_token = st.secrets["HF_TOKEN"]
        self._client = Client(self.space_id, hf_token=hf_token)

    def _call(self, prompt: str, **kwargs) -> str:
        result = self._client.predict(
            message=prompt,
            history=[],
            use_deep_research=False,
            api_name="/query_deepseek_streaming"
        )
        #print("RAW RESULT:", result)

        # Extract the model-generated answer from the result
        try:
            if isinstance(result, (list, tuple)):
                return result[0][0][1]  # Actual answer
            elif isinstance(result, str):
                return result
            else:
                return "Unexpected response format."
        except Exception as e:
            return f"Error extracting response: {e}"
    
    @property
    def _llm_type(self) -> str:
        return "gradio-mistral"

#---------------------------------------------------------------------------
