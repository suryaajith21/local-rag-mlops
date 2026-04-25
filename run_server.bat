@echo off
conda activate rag-ops
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
