"""Drive eval/test_queries.json through the running API to populate the runs table.

Posts each query to /runs, then polls /runs/{job_id} until done/failed before
moving on (one at a time to respect Groq/arXiv rate limits).
"""
import json
import os
import time

import requests

BASE = os.environ.get("PAPERPULSE_API", "http://127.0.0.1:8000")
HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "test_queries.json")) as f:
    queries = json.load(f)

for i, q in enumerate(queries, 1):
    print(f"\n[{i}/{len(queries)}] {q}")
    job_id = requests.post(f"{BASE}/runs", json={"query": q}).json()["job_id"]
    print(f"  job_id={job_id} ", end="", flush=True)

    while True:
        time.sleep(10)
        job = requests.get(f"{BASE}/runs/{job_id}").json()
        status = job["status"]
        if status == "running":
            print(".", end="", flush=True)
            continue
        if status == "done":
            print(f"\n  done — eval_score={job.get('eval_score')} "
                  f"iterations={job.get('iterations')}")
        else:
            print(f"\n  FAILED — {job.get('error')}")
        break

print("\nAll queries processed.")
