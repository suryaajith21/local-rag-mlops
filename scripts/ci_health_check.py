import httpx, sys

try:
    r = httpx.get('http://localhost:8000/health', timeout=5.0)
    data = r.json()
    assert data.get('status') == 'ok', f'Bad status: {data}'
    assert data.get('vector_chunks', 0) > 0, f'No chunks: {data}'
    print(f'OK: server healthy, {data["vector_chunks"]} chunks')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
