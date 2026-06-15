import sys
import os
import json
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import orchestrator_run

with open("eval/test_queries.json") as f:
    queries = json.load(f)

results = []
for i, query in enumerate(queries):
    print(f"\n{'='*50}")
    print(f"Query {i+1}/{len(queries)}: {query}")
    print('='*50)
    try:
        state = orchestrator_run(query)
        results.append({
            "query": query,
            "overall": state["eval_score"]["overall"],
            "passed": state["eval_score"]["passed"],
            "iterations": state["iterations"]
        })
        print(f"✓ overall={state['eval_score']['overall']} passed={state['eval_score']['passed']}")
    except Exception as e:
        print(f"✗ failed: {e}")
        results.append({"query": query, "error": str(e)})
    
    time.sleep(30)  

print("\n=== BATCH RESULTS ===")
passed = [r for r in results if r.get("passed")]
print(f"Passed: {len(passed)}/{len(results)}")
avg = sum(r["overall"] for r in results if "overall" in r) / len(results)
print(f"Average overall score: {avg:.3f}")

with open("eval/batch_results.json", "w") as f:
    json.dump(results, f, indent=2)