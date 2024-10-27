from pydantic import BaseModel
from app.utils.gpt_models import GPTModel


class GPTBasicTextRequest(BaseModel):
    text: str
    model: GPTModel = GPTModel.GPT_4


class GPTBasicTextResponse(BaseModel):
    summary: str
