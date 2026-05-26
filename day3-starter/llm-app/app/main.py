import weave
from dotenv import load_dotenv
from fastapi import FastAPI
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
    run_id: str | None


@app.post("/api/generate")
@weave.op()
async def generate(req: GenerateRequest) -> GenerateResponse:
    result = await graph.ainvoke(
        {
            "customer_name": req.customer_name,
            "company_name": req.company_name,
            "content": req.content,
        },
    )
    call = weave.get_current_call()
    weave_call_id = str(call.id) if call is not None else None

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
