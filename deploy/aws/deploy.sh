#!/usr/bin/env bash
# Industry path: Docker image → ECR → ECS Fargate + ALB + Secrets Manager.
# Region defaults to Frankfurt (eu-central-1).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

REGION="${AWS_REGION:-eu-central-1}"
REPO="${ECR_REPO:-materials-agent}"
STACK="${STACK_NAME:-materials-intelligence}"
SECRET_NAME="${SECRET_NAME:-materials-intelligence/groq}"
TAG="${IMAGE_TAG:-$(date +%Y%m%d%H%M%S)}"

need() { command -v "$1" >/dev/null || { echo "Missing $1"; exit 1; }; }
need aws
need docker

if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is not running. Open it and retry."
  exit 1
fi

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Missing .env with GROQ_API_KEY"
  exit 1
fi
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a
if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "GROQ_API_KEY is empty in .env"
  exit 1
fi

ACCOUNT="$(aws sts get-caller-identity --query Account --output text --region "$REGION")"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

echo "Account $ACCOUNT"
echo "Region  $REGION"
echo "Image   $IMAGE"

VPC="$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
if [[ -z "$VPC" || "$VPC" == "None" ]]; then
  echo "No default VPC in $REGION. Create one in the AWS console, then retry."
  exit 1
fi
SUBNETS="$(aws ec2 describe-subnets --region "$REGION" \
  --filters Name=vpc-id,Values="$VPC" Name=map-public-ip-on-launch,Values=true \
  --query 'Subnets[].SubnetId' --output text | tr '\t' ',')"
SUBNET_COUNT="$(echo "$SUBNETS" | awk -F, '{print NF}')"
if [[ "$SUBNET_COUNT" -lt 2 ]]; then
  echo "Need at least two public subnets in $VPC"
  exit 1
fi
echo "VPC     $VPC"
echo "Subnets $SUBNETS"

if ! aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null
  echo "Created ECR repository $REPO"
fi

if SECRET_ARN="$(aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$REGION" --query ARN --output text 2>/dev/null)"; then
  aws secretsmanager put-secret-value --secret-id "$SECRET_NAME" --region "$REGION" \
    --secret-string "$GROQ_API_KEY" >/dev/null
  echo "Updated secret $SECRET_NAME"
else
  SECRET_ARN="$(aws secretsmanager create-secret --name "$SECRET_NAME" --region "$REGION" \
    --secret-string "$GROQ_API_KEY" --query ARN --output text)"
  echo "Created secret $SECRET_NAME"
fi

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

echo "Building linux/amd64 image…"
docker build --platform linux/amd64 -t "$REPO:$TAG" .
docker tag "$REPO:$TAG" "$IMAGE"
docker push "$IMAGE"

PARAM_FILE="$(mktemp)"
python3 - "$PARAM_FILE" "$IMAGE" "$SECRET_ARN" "$VPC" "$SUBNETS" <<'PY'
import json, sys
path, image, secret, vpc, subnets = sys.argv[1:]
json.dump(
    [
        {"ParameterKey": "ImageUri", "ParameterValue": image},
        {"ParameterKey": "GroqSecretArn", "ParameterValue": secret},
        {"ParameterKey": "VpcId", "ParameterValue": vpc},
        {"ParameterKey": "SubnetIds", "ParameterValue": subnets},
    ],
    open(path, "w"),
)
PY

if aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" >/dev/null 2>&1; then
  echo "Updating CloudFormation stack $STACK"
  aws cloudformation update-stack --stack-name "$STACK" --region "$REGION" \
    --template-body "file://${ROOT}/deploy/aws/ecs.yml" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters "file://${PARAM_FILE}" >/dev/null
  aws cloudformation wait stack-update-complete --stack-name "$STACK" --region "$REGION"
else
  echo "Creating CloudFormation stack $STACK"
  aws cloudformation create-stack --stack-name "$STACK" --region "$REGION" \
    --template-body "file://${ROOT}/deploy/aws/ecs.yml" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters "file://${PARAM_FILE}" >/dev/null
  aws cloudformation wait stack-create-complete --stack-name "$STACK" --region "$REGION"
fi
rm -f "$PARAM_FILE"

URL="$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='Url'].OutputValue" --output text)"
echo
echo "Live URL: $URL"
echo "Logs:     CloudWatch → /ecs/materials-intelligence"
echo "Stop bill: aws cloudformation delete-stack --stack-name $STACK --region $REGION"
