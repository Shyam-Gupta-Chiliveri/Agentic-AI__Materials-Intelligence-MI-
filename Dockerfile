FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-agent.txt .
RUN pip install --no-cache-dir -r requirements-agent.txt

COPY agent/ agent/
COPY apps/ apps/
COPY .streamlit/ .streamlit/
COPY data/motor/ data/motor/
COPY faiss_index_local/ faiss_index_local/

# Only the small axle models the agent loads (skip 1.3 GB unused pickles).
COPY models/best_ductility_model.pkl models/best_brittleness_model.pkl \
     models/scaler.pkl models/le_die_casting.pkl models/le_diameter.pkl \
     models/

# Smaller SEM checkpoint; classification is skipped if this file is absent.
COPY sem_output/models/sem_classifier_final.pth sem_output/models/

ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "streamlit run apps/agentic_materials_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
