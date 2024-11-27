from app.schemas.redditbias import Topic
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from typing import Literal

from app.utils.reddibias_evaluation import redditbias_measure_bias


router = APIRouter(
    prefix="/reddit-bias/evaluation",
    tags=["Reddit Bias: Evaluation"],
)


@router.post("/measure-bias/{topic}")
def get_raw_comments_from_topic(
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"],
    on_set: bool = True,
    get_perplexity: bool = True,
    reduce_set: bool = False,
):
    """
    Performs Student t-test on the perplexity distribution of two sentences groups
    with contrasting targets
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Get and store the raw reddit comments
        redditbias_measure_bias(
            topic_ins=topic_instance,
            on_set=on_set,
            get_perplexity=get_perplexity,
            reduce_set=reduce_set,
        )

        return {"message": "The comments were successfully evaluated"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
