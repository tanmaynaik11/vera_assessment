from input_classifier import classify

TEST_QUERIES = [
    {
        "id": 1,
        "query": "What is the first-line treatment for hypertension in a 55-year-old male with diabetes?",
        "context": "ED, inpatient",
        "expected": "PATIENT_SPECIFIC",
    },
    {
        "id": 2,
        "query": "Severe UC flare in a 32F with prior steroid response",
        "context": "GI, inpatient",
        "expected": "PATIENT_SPECIFIC",
    },
    {
        "id": 3,
        "query": "Differential for 45F with acute SOB, pleuritic CP, unilateral leg swelling, recent travel",
        "context": "ED, must-not-miss",
        "expected": "PATIENT_SPECIFIC",
    },
    {
        "id": 4,
        "query": "Refractory GERD after failed BID PPI in a 45M",
        "context": "GI, outpatient",
        "expected": "PATIENT_SPECIFIC",
    },
    {
        "id": 5,
        "query": "New-onset AFib with RVR in a healthy 65F",
        "context": "cardiology, acute",
        "expected": "PATIENT_SPECIFIC",
    },
    {
        "id": 6,
        "query": "What are the latest guidelines for secondary stroke prevention",
        "context": "general",
        "expected": "GUIDELINE_ONLY",
    },
]

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def run_tests():
    print("=" * 75)
    print(f"{'#':<4} {'RESULT':<8} {'GOT':<20} {'CONF':<8} {'CONTEXT':<22} QUERY")
    print("=" * 75)

    passed = 0
    results = []

    for t in TEST_QUERIES:
        result = classify(t["query"])
        got = result["classification"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]
        matched = got == t["expected"]
        status = PASS if matched else FAIL
        if matched:
            passed += 1

        print(f"{t['id']:<4} {status:<17} {got:<20} {confidence:<8} {t['context']:<22} {t['query'][:45]}...")
        print(f"     Reasoning: {reasoning}")
        print()

        results.append({**t, "got": got, "confidence": confidence, "reasoning": reasoning, "passed": matched})

    print("=" * 75)
    print(f"Results: {passed}/{len(TEST_QUERIES)} passed")
    return results


if __name__ == "__main__":
    run_tests()
