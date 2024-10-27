from fastapi import APIRouter, HTTPException
from app.schemas.gpt_text import GPTBasicTextResponse, GPTBasicTextRequest
from app.external_services.gpt_service import rewrite_document

router = APIRouter(
    prefix="/gpt",
    tags=["GPT Text Operations"],
)


@router.post(
    "/rewrite-style",
    response_model=GPTBasicTextResponse,
    summary="Summarize a text using GPT",
)
async def rewrite_text(request: GPTBasicTextRequest):
    try:
        summary = await rewrite_document(request.text)
        return GPTBasicTextResponse(summary=summary)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error rewriting the text: {str(e)}"
        )
