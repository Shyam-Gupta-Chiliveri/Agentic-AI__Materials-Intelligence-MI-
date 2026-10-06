#!/bin/bash
# Install k3s (no Traefik — Caddy already owns 80/443), import the app image, apply manifests.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v k3s >/dev/null 2>&1; then
  curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik --write-kubeconfig-mode 644" sh -
fi

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

echo "Importing materials-agent:latest into k3s (a few minutes)..."
sudo docker save materials-agent:latest | sudo k3s ctr images import -

KEY=$(sudo docker inspect materials-app --format '{{range .Config.Env}}{{println .}}{{end}}' | awk -F= '/^GROQ_API_KEY=/{print $2}')
GRAFANA_PW=$(openssl rand -hex 12)

sudo kubectl apply -f k8s/namespaces.yaml
sudo kubectl -n materials create secret generic groq --from-literal=GROQ_API_KEY="$KEY" --dry-run=client -o yaml | sudo kubectl apply -f -
sudo kubectl -n monitoring create secret generic grafana-admin --from-literal=password="$GRAFANA_PW" --dry-run=client -o yaml | sudo kubectl apply -f -
sudo kubectl apply -f k8s/apps.yaml
sudo kubectl apply -f k8s/monitoring.yaml
sudo kubectl apply -f k8s/grafana.yaml

echo "Waiting for app pods..."
sudo kubectl -n materials rollout status deploy/agentic --timeout=180s
sudo kubectl -n materials rollout status deploy/sem --timeout=180s
sudo kubectl -n materials rollout status deploy/rag --timeout=180s
sudo kubectl -n monitoring rollout status deploy/grafana --timeout=180s

echo "Switching Caddy from Docker ports to Kubernetes NodePorts..."
sudo tee /etc/caddy/Caddyfile >/dev/null << 'EOF'
138.2.179.154.sslip.io {
    reverse_proxy localhost:30080
}
sem.138.2.179.154.sslip.io {
    reverse_proxy localhost:30082
}
rag.138.2.179.154.sslip.io {
    reverse_proxy localhost:30083
}
grafana.138.2.179.154.sslip.io {
    reverse_proxy localhost:30300
}
EOF
sudo systemctl reload caddy

echo "Stopping Docker copies so only Kubernetes serves the apps..."
sudo docker stop materials-app sem-app rag-app || true

echo "Grafana admin password (store privately): $GRAFANA_PW"
echo "Viewer URL (no login): https://grafana.138.2.179.154.sslip.io"
sudo kubectl get pods -A
