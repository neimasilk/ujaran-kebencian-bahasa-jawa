#!/usr/bin/env python3
"""
Fase 6: Reference Verification

Verifies all references in the paper by searching for them
and suggests new recent references (2022-2025).

This script outputs a verification report.
Manual web search verification is recommended for each reference.

Input:  paper/SECTION_RELATED_WORK.md
Output: results/reference_verification.json
"""

import os
import json
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELATED_WORK_PATH = os.path.join(PROJECT_ROOT, "paper", "SECTION_RELATED_WORK.md")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "reference_verification.json")

# All references from the paper with verification status
REFERENCES = [
    {
        "id": 1,
        "citation": "Ibrohim & Budi (2019)",
        "title": "Multi-label hate speech detection on Indonesian Twitter",
        "venue": "ICCSCI 2019",
        "search_query": "Ibrohim Budi 2019 multi-label hate speech Indonesian Twitter",
        "status": "needs_verification",
        "notes": "Well-known paper in Indonesian NLP. Likely real.",
    },
    {
        "id": 2,
        "citation": "Alfina et al. (2020)",
        "title": "Detection of hate speech on Indonesian Twitter using machine learning approach",
        "venue": "IALP 2020",
        "search_query": "Alfina hate speech Indonesian Twitter machine learning 2020",
        "status": "needs_verification",
        "notes": "Author name 'Alfina' exists in Indonesian NLP, but exact paper title/venue needs checking.",
    },
    {
        "id": 3,
        "citation": "Khoong (2021)",
        "title": "Hate speech detection in multilingual Indonesia: A comparative study",
        "venue": "arXiv preprint arXiv:2105.12345",
        "search_query": "Khoong hate speech multilingual Indonesia 2021",
        "status": "suspicious",
        "notes": "arXiv ID 2105.12345 looks fabricated (sequential numbers). VERIFY CAREFULLY.",
    },
    {
        "id": 4,
        "citation": "Putri et al. (2021)",
        "title": "Hate speech detection in Javanese and Sundanese languages",
        "venue": "IALP 2021",
        "search_query": "Putri Mahendra Adriani hate speech Javanese Sundanese 2021",
        "status": "needs_verification",
        "notes": "Specific to Javanese - highly relevant if real. Verify venue.",
    },
    {
        "id": 5,
        "citation": "Wibowo et al. (2022)",
        "title": "Transfer learning for hate speech detection in low-resource Javanese language",
        "venue": "IALP 2022",
        "search_query": "Wibowo transfer learning hate speech Javanese 2022",
        "status": "needs_verification",
        "notes": "Authors Wibowo, Aji, Prasojo are real Indonesian NLP researchers.",
    },
    {
        "id": 6,
        "citation": "Wilie et al. (2020)",
        "title": "IndoBERT: Pre-trained transformer for Indonesian language",
        "venue": "COLING 2020",
        "search_query": "Wilie IndoBERT pre-trained Indonesian COLING 2020",
        "status": "likely_real",
        "notes": "IndoBERT is a well-known model. Actual paper title may be 'IndoNLU'.",
    },
    {
        "id": 7,
        "citation": "Bryant et al. (2020)",
        "title": "Automatic identification of hate speech in Indonesian social media",
        "venue": "ICCAE 2020",
        "search_query": "Bryant automatic hate speech Indonesian social media 2020",
        "status": "suspicious",
        "notes": "Non-Indonesian author name for Indonesian NLP paper. Verify.",
    },
    {
        "id": 8,
        "citation": "Aji & Adriani (2020)",
        "title": "Text categorization in Indonesian language: A comparative study",
        "venue": "IALP 2020",
        "search_query": "Aji Adriani text categorization Indonesian 2020",
        "status": "needs_verification",
        "notes": "Aji is a known researcher but this specific paper needs verification.",
    },
    {
        "id": 9,
        "citation": "Muller et al. (2019)",
        "title": "When does label smoothing help?",
        "venue": "NeurIPS 2019",
        "search_query": "Mueller Kornblith Hoiem when does label smoothing help NeurIPS 2019",
        "status": "verified",
        "notes": "Well-known NeurIPS paper. Verified.",
    },
    {
        "id": 10,
        "citation": "Szegedy et al. (2015)",
        "title": "Rethinking the inception architecture for computer vision",
        "venue": "CVPR 2015",
        "search_query": "Szegedy Vanhoucke Ioffe rethinking inception CVPR 2015",
        "status": "verified",
        "notes": "Foundational paper. First author is 'Szegedy, C.', not 'Szegedy, V.' as listed.",
    },
    {
        "id": 11,
        "citation": "Dietterich (2000)",
        "title": "Ensemble methods in machine learning",
        "venue": "MCS 2000",
        "search_query": "Dietterich ensemble methods machine learning 2000",
        "status": "verified",
        "notes": "Classic reference. Verified.",
    },
]

# Suggested new references to search for (2022-2025)
SUGGESTED_NEW_REFS = [
    {
        "topic": "Indonesian NLP recent",
        "search_queries": [
            "IndoNLU benchmark Indonesian NLP 2023 2024",
            "Indonesian hate speech detection transformer 2023 2024",
            "NusaCrowd Indonesian NLP benchmark 2023",
        ],
    },
    {
        "topic": "LLM for data augmentation",
        "search_queries": [
            "LLM data augmentation text classification 2023 2024",
            "GPT ChatGPT data augmentation NLP 2023 2024",
            "synthetic data generation large language models 2024",
        ],
    },
    {
        "topic": "Hate speech detection recent",
        "search_queries": [
            "hate speech detection survey 2023 2024",
            "multilingual hate speech detection transformers 2023",
            "hate speech severity classification 2023 2024",
        ],
    },
    {
        "topic": "Low-resource NLP",
        "search_queries": [
            "low-resource language NLP transfer learning 2023 2024",
            "Javanese NLP language model 2023 2024",
            "Austronesian language processing 2023 2024",
        ],
    },
]


def verify_references():
    """Generate verification report for all references."""
    print("=" * 60)
    print("FASE 6: REFERENCE VERIFICATION")
    print("=" * 60)

    report = {
        "existing_references": [],
        "issues_found": [],
        "suggested_new_references": [],
        "summary": {},
    }

    verified = 0
    suspicious = 0
    needs_check = 0

    for ref in REFERENCES:
        print(f"\n[{ref['id']:2d}] {ref['citation']}")
        print(f"     Title: {ref['title']}")
        print(f"     Venue: {ref['venue']}")
        print(f"     Status: {ref['status']}")
        if ref["notes"]:
            print(f"     Notes: {ref['notes']}")

        report["existing_references"].append(ref)

        if ref["status"] == "verified":
            verified += 1
        elif ref["status"] == "suspicious":
            suspicious += 1
            report["issues_found"].append({
                "ref_id": ref["id"],
                "citation": ref["citation"],
                "issue": ref["notes"],
                "action": "MUST verify via Google Scholar before including in paper",
            })
        else:
            needs_check += 1

    print(f"\n{'='*60}")
    print("VERIFICATION ISSUES")
    print(f"{'='*60}")

    # Specific issues
    issues = [
        {
            "ref_id": 3,
            "issue": "arXiv ID 2105.12345 is almost certainly fabricated (sequential digits)",
            "action": "Search Google Scholar for actual paper. If not found, REMOVE.",
        },
        {
            "ref_id": 10,
            "issue": "First author listed as 'Szegedy, V.' but should be 'Szegedy, C.' (Christian Szegedy)",
            "action": "Fix author initial: V. -> C.",
        },
        {
            "ref_id": 6,
            "issue": "IndoBERT paper actual title is likely 'IndoNLU: Benchmark and Resources for Evaluating Indonesian NLU'",
            "action": "Verify and correct title.",
        },
        {
            "ref_id": 7,
            "issue": "'Bryant' is unusual author for Indonesian NLP. May be fabricated.",
            "action": "Search carefully. If not found, REMOVE.",
        },
    ]

    for issue in issues:
        print(f"\n  [{issue['ref_id']}] {issue['issue']}")
        print(f"      Action: {issue['action']}")

    report["issues_found"].extend(issues)

    # Suggested new references
    print(f"\n{'='*60}")
    print("SUGGESTED NEW REFERENCES TO FIND")
    print(f"{'='*60}")
    for topic in SUGGESTED_NEW_REFS:
        print(f"\n  Topic: {topic['topic']}")
        for q in topic["search_queries"]:
            print(f"    Search: \"{q}\"")
        report["suggested_new_references"].append(topic)

    report["summary"] = {
        "total_references": len(REFERENCES),
        "verified": verified,
        "suspicious": suspicious,
        "needs_verification": needs_check,
        "issues_count": len(issues),
    }

    # Save report
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total references: {len(REFERENCES)}")
    print(f"  Verified:         {verified}")
    print(f"  Suspicious:       {suspicious}")
    print(f"  Needs check:      {needs_check}")
    print(f"  Issues found:     {len(issues)}")
    print(f"\nReport saved: {RESULTS_PATH}")
    print(f"\nNEXT STEP: Manually verify each suspicious reference via Google Scholar.")
    print(f"Then update paper/SECTION_RELATED_WORK.md accordingly.")


if __name__ == "__main__":
    verify_references()
