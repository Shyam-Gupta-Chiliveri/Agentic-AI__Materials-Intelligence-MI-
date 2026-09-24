# Agentic AI — Materials Intelligence

A case desk for steel axle fractures. Give it a hardness value, a heat-treatment question, or an SEM image and it works the case through to a decision: how the axle is likely to fail, which process step is driving that, and what to verify next on the line.

The answer is not a pile of tool output. Standards, the process graph, the fracture model, the SEM classifier, and the motor session are brought onto one case. If two signals disagree, the case says so, then still closes with a recommendation an engineer can use.

**Live demo:** [https://d1odeaab3kt4gn.cloudfront.net](https://d1odeaab3kt4gn.cloudfront.net)

**Phases 1–4** (EDA, machine learning, SEM U-Net, RAG) are in a separate repository:

[AI-based material analysis system with multimodal learning](https://github.com/Shyam-Gupta-Chiliveri/AI-based-material-analysis-system-with-multimodal-learning)

Deployed on AWS ECS Fargate behind CloudFront, region `eu-central-1` (Frankfurt).

## How a case is closed

1. The question is sent only to the specialists that case needs.
2. Each specialist returns a result the case can stand on: the relevant standard, the heat-treatment path, ductile and brittle share, the SEM split, an HV10 band, and the motor session.
3. Those results are written as one engineering answer: what is happening, why the process produced it, and the next check. Higher tempering temperature lowers hardness and raises ductility; the case follows that, it does not invent a second story.

## Specialists on the case

| Specialist | What it settles |
|---|---|
| Standards | The ISO/DIN clause that governs the question |
| Knowledge | Heat treatment, hardness, fracture mode, and SEM, linked as one process |
| Fracture | Ductile % and brittle % from HV10 |
| Vision | Ductile versus brittle area on the SEM image, then an HV10 band from that split |
| Twin | Whether this plant motor session is inside its normal window |
| Critic | A clear flag when the axle model and the SEM do not agree |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-agent.txt
# put GROQ_API_KEY in .env — never commit that file
streamlit run apps/agentic_materials_app.py --server.port 8505
```

```bash
PYTHONPATH=. pytest -q
```

## Deploy

See [DEPLOY.md](DEPLOY.md). Production path: Docker image → ECR → ECS Fargate + ALB + Secrets Manager.

```bash
docker build --platform linux/amd64 -t materials-agent .
docker run -p 8080:8080 --env-file .env materials-agent
```

Model weights (`models/*.pkl`, `sem_output/models/sem_classifier_final.pth`) stay out of git because of size. The running AWS image already includes them. Place the same files locally before a Docker build.

## Layout

```
agent/          router, specialists, tools, knowledge graph
apps/           Streamlit app
tests/          pytest
deploy/aws/     CloudFormation + deploy script
faiss_index_local/   standards index
data/motor/     plant motor session summaries
```
