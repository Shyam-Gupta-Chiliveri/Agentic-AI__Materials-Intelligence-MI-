# Agentic AI — Materials Intelligence

Multi-agent system for steel axle fracture questions. Deterministic tools return the numbers. The language model only writes the answer from those tool results.

**Live demo:** [https://d1odeaab3kt4gn.cloudfront.net](https://d1odeaab3kt4gn.cloudfront.net)

**Phases 1–4** (EDA, machine learning, SEM U-Net, RAG) are in a separate repository:

[AI-based material analysis system with multimodal learning](https://github.com/Shyam-Gupta-Chiliveri/AI-based-material-analysis-system-with-multimodal-learning)

Deployed on AWS ECS Fargate behind CloudFront, region `eu-central-1` (Frankfurt).

## What it does

| Agent | Tool | Job |
|---|---|---|
| Standards | `search_iso_standards` | FAISS search over ISO/DIN text |
| Knowledge | `query_knowledge_graph` | Process links: heat treatment, hardness, fracture, SEM |
| Fracture | `predict_axle_fracture` | Ridge model for ductile % and brittle % from HV10 |
| Vision | `classify_sem_image` | U-Net pixel split of an SEM fracture image |
| Vision | `estimate_hv10_from_sem` | HV10 range from the SEM ductile % |
| Twin | `assess_motor_session` | Plant motor session check |
| Critic | `detect_conflict` | Flags when axle and SEM brittle % disagree |

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
