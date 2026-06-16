import json
from agents.evaluator_agent import evaluator_agent_run

# Query, paper, and digest all aligned on protein folding now.
query = "vLLMs for long-context language modeling"

papers = [
    {
        "id": "2401.99999",
        "title": "AlphaFold-Style Deep Learning for Protein Structure Prediction",
    }
]

extractions = {
    "2401.99999": {
        "methodology": "A graph neural network trained on the PDB to predict 3D protein "
                       "structures from amino-acid sequences.",
        "key_claims": [
            "Predicts protein backbone structure to near-experimental accuracy.",
            "Outperforms prior physics-based folding simulations on CASP targets.",
        ],
        "results": "Achieves a median GDT-TS of 92 on CASP14 free-modeling targets.",
        "limitations": [
            "Struggles with intrinsically disordered regions.",
            "Requires large multiple-sequence alignments at inference time.",
        ],
    }
}

# Digest restates the extraction's own claims/results — fully grounded.
digest = {
    "summary": "Deep learning has reached near-experimental accuracy for protein structure "
               "prediction. A graph neural network trained on the PDB predicts 3D structure "
               "directly from amino-acid sequence, outperforming prior physics-based folding "
               "simulations, though it still struggles with disordered regions.",
    "key_findings": [
        "A graph neural network trained on the PDB predicts protein backbone structure to "
        "near-experimental accuracy.",
        "The method outperforms prior physics-based folding simulations on CASP targets, "
        "achieving a median GDT-TS of 92 on CASP14 free-modeling targets.",
    ],
    "contradictions": [],
    "recommended_papers": [
        {
            "paper_id": "2401.99999",
            "title": "AlphaFold-Style Deep Learning for Protein Structure Prediction",
            "why": "Directly addresses the query with state-of-the-art structure prediction.",
        }
    ],
    "what_to_read_next": "Investigate handling of intrinsically disordered regions and "
                         "reducing reliance on large multiple-sequence alignments.",
}

result = evaluator_agent_run(query, digest, papers, extractions)
print("\n=== EVALUATION ===")
print(json.dumps(result["evaluation"], indent=2))
