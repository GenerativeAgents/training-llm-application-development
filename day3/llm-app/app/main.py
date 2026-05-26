import weave
from dotenv import load_dotenv
from fastapi import FastAPI
from openevals.string.levenshtein import levenshtein_distance
from pydantic import BaseModel

from app.generate.graph import graph
from app.generate.types import QualityJudgment, TopicType

load_dotenv()
weave_client = weave.init("training-ai-agent-dev")

app = FastAPI(
    title="LLM App",
    description="AI response generation for inquiry management",
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


class GenerateRequest(BaseModel):
    customer_name: str
    company_name: str | None = None
    content: str


class QualityScores(BaseModel):
    politeness: QualityJudgment
    politeness_reason: str


class GeneratedDraft(BaseModel):
    subject: str
    body: str
    quality_scores: QualityScores


class GenerateResponse(BaseModel):
    topic: TopicType
    classification_confidence: float
    generated_draft: GeneratedDraft | None
    weave_call_id: str | None


@app.post("/api/generate")
@weave.op()
async def generate(req: GenerateRequest) -> GenerateResponse:
    call = weave.require_current_call()
    weave_call_id = str(call.id)

    result = await graph.ainvoke(
        {
            "customer_name": req.customer_name,
            "company_name": req.company_name,
            "content": req.content,
        },
    )

    topic = result["topic"]
    classification_confidence = result["classification_confidence"]

    if topic == "spam":
        generated_draft = None
    else:
        generated_draft = GeneratedDraft(
            subject=result["response_subject"],
            body=result["response_body"],
            quality_scores=QualityScores(
                politeness=result["politeness"],
                politeness_reason=result["politeness_reason"],
            ),
        )

    return GenerateResponse(
        topic=topic,
        classification_confidence=classification_confidence,
        generated_draft=generated_draft,
        weave_call_id=weave_call_id,
    )


class FeedbackRequest(BaseModel):
    weave_call_id: str
    ai_body: str
    final_body: str
    original_topic: str
    current_topic: str


class FeedbackResponse(BaseModel):
    operator_edited_topic: bool
    edit_distance: float


@app.post("/api/feedback")
async def post_feedback(req: FeedbackRequest) -> FeedbackResponse:
    call = weave_client.get_call(req.weave_call_id)

    # operator_edited_topic: 編集なし (AI正解)=1.0、編集あり=0.0
    edited = req.original_topic != req.current_topic
    topic_score = 0.0 if edited else 1.0
    call.feedback.add("operator_edited_topic", {"value": topic_score})

    # openevals で edit_distance を算出 (1.0=完全一致、0.0=完全不一致)
    result = levenshtein_distance(outputs=req.final_body, reference_outputs=req.ai_body)
    edit_score = result["score"]
    call.feedback.add("edit_distance", {"value": edit_score})

    return FeedbackResponse(operator_edited_topic=edited, edit_distance=edit_score)
