# Materials Intelligence platform (Terraform + Kubernetes)

This is **not a new chatbot**. It is the production platform around the three Streamlit apps you already have:

| App | Public URL |
|---|---|
| Agentic AI | https://138.2.179.154.sslip.io |
| SEM classifier | https://sem.138.2.179.154.sslip.io |
| RAG | https://rag.138.2.179.154.sslip.io |
| **Grafana (this project)** | https://grafana.138.2.179.154.sslip.io |

**Terraform** describes the Oracle Cloud VM and firewall. **k3s** (Kubernetes) runs the apps. **Prometheus** scrapes health. **Grafana** is the dashboard you open in a browser.

Single-node Kubernetes on Oracle Always Free (Ampere ARM, Frankfurt). Cost: **$0**.

## Resume bullets

- Provisioned an Oracle Cloud Always Free ARM VM with Terraform-ready IaC (VCN, security lists, VM.Standard.A1.Flex).
- Deployed a three-service agentic materials platform on k3s (Kubernetes): Agentic AI, SEM U-Net classifier, ISO/DIN RAG.
- Added Prometheus + Grafana with blackbox probes of the public HTTPS demos and node CPU/memory dashboards.

## Layout

```
deploy/platform/
  terraform/     OCI VM, network, ports 22/80/443
  k8s/           Deployments, Services, Prometheus, Grafana
  scripts/       k3s install + image import + kubectl apply
```

## Live cluster (already applied on the demo VM)

```bash
ssh ubuntu@138.2.179.154
sudo kubectl get pods -A
```

Open Grafana as **Viewer** (no login required for the default dashboard).

## Recreate from zero

1. Fill `terraform/terraform.tfvars` from the example (OCI tenancy, key, subnet).
2. `cd terraform && terraform init && terraform apply`
3. SSH to the new VM and run `scripts/bootstrap.sh`

Do **not** `terraform apply` against an existing Always Free VM unless you intend to replace it. The live demo was bootstrapped with `scripts/bootstrap.sh` on the VM that already hosted Docker.
