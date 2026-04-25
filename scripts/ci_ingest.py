import asyncio
from src.ingestion.pipeline import run_ingestion_pipeline

summary = asyncio.run(run_ingestion_pipeline(clear_existing=True))
print('Ingestion summary:')
for k, v in summary.items():
    print(f'  {k}: {v}')
files = summary.get('files_processed', 0)
assert files > 0, f'No files ingested - check data/ directory'
print(f'OK: {files} files ingested')
