# Deploy the agentic app

Keep `GROQ_API_KEY` in secrets. Never commit `.env`.

## Local

```bash
streamlit run apps/agentic_materials_app.py --server.port 8505
```

## Docker

```bash
docker build --platform linux/amd64 -t materials-agent .
docker run -p 8080:8080 --env-file .env materials-agent
```

The image copies FAISS, motor summaries, the small axle `.pkl` models, and `sem_classifier_final.pth`. The 1.3 GB unused `*_optimized.pkl` files stay out.

## Streamlit Community Cloud

1. Push this repo to GitHub (large PDFs stay out of git).
2. New app → `apps/agentic_materials_app.py`.
3. Add secret `GROQ_API_KEY`.
4. Set Python path so `agent/` imports: add `PYTHONPATH=.` in the Cloud advanced settings, or keep `apps/` as the working directory and leave `sys.path` as in the app.

## Hugging Face Spaces

1. New Space → Streamlit SDK.
2. Upload `apps/`, `agent/`, `models/`, `data/motor/`, `faiss_index_local/`, `.streamlit/`.
3. Space secret: `GROQ_API_KEY`.

## AWS — ECS Fargate (industry path)

Same pattern used on AWS accounts at BMW / Bosch / Siemens / VW-style platforms: **Docker image → ECR → ECS Fargate behind an ALB**, secret in **Secrets Manager**, region **`eu-central-1` (Frankfurt)**.

```bash
aws configure          # access key, secret, region eu-central-1, output json
chmod +x deploy/aws/deploy.sh
./deploy/aws/deploy.sh
```

The script creates or updates:

| Piece | Name | Role |
|-------|------|------|
| ECR | `materials-agent` | Immutable tagged image |
| Secrets Manager | `materials-intelligence/groq` | `GROQ_API_KEY` (not in the image) |
| ECS cluster | `materials-intelligence` | Fargate, 2 vCPU / 4 GB |
| ALB | `materials-intelligence` | Public URL, health check `/_stcore/health` |
| CloudWatch | `/ecs/materials-intelligence` | Container logs |
| CloudFormation | `materials-intelligence` | Whole stack, one delete to stop cost |

A plant production setup would add HTTPS (ACM + Route 53), private subnets + NAT, and company SSO. This stack is the public-demo version of that architecture.

Tear down when the demo is over:

```bash
aws cloudformation delete-stack --stack-name materials-intelligence --region eu-central-1
```

## Azure (same container)

Push to ACR, run on Container Apps or App Service, port **8080**, secret `GROQ_API_KEY`.

The unit-test CI workflow (`.github/workflows/ci.yml`) is the CD gate before you ship the image.
