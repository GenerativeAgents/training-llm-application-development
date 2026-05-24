from fastapi import FastAPI
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from openevals.string.levenshtein import levenshtein_distance
from pydantic import BaseModel

from app.generate.graph import graph
from app.generate.types import QualityJudgment, TopicType

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
    run_id: str | None


@app.post("/api/generate")
@traceable
async def generate(req: GenerateRequest) -> GenerateResponse:
    result = await graph.ainvoke(
        {
            "customer_name": req.customer_name,
            "company_name": req.company_name,
            "content": req.content,
        },
    )
    rt = get_current_run_tree()
    run_id = str(rt.id) if rt is not None else None

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
        run_id=run_id,
    )


class FeedbackRequest(BaseModel):
    run_id: str
    ai_body: str
    final_body: str
    original_topic: str
    current_topic: str


class FeedbackResponse(BaseModel):
    operator_edited_topic: bool
    edit_distance: float


@app.post("/api/feedback")
async def post_feedback(req: FeedbackRequest) -> FeedbackResponse:
    client = Client()

    # operator_edited_topic: 編集なし (AI正解)=1.0、編集あり=0.0
    edited = req.original_topic != req.current_topic
    topic_score = 0.0 if edited else 1.0
    client.create_feedback(
        run_id=req.run_id,
        key="operator_edited_topic",
        score=topic_score,
    )

    # openevals で edit_distance を算出 (1.0=完全一致、0.0=完全不一致)
    result = levenshtein_distance(outputs=req.final_body, reference_outputs=req.ai_body)
    edit_score = result["score"]
    client.create_feedback(
        run_id=req.run_id,
        key="edit_distance",
        score=edit_score,
    )

    return FeedbackResponse(operator_edited_topic=edited, edit_distance=edit_score)
