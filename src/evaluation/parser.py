import re

def parse_judge_scores(judge_response):
    correctness_match = re.search(r"Correctness:\s*(\d)", judge_response)
    completeness_match =  re.search(r"Completeness:\s*(\d)", judge_response)
    fluency_match =  re.search(r"Fluency:\s*(\d)", judge_response)

    correctness = (int(correctness_match.group(1)) if correctness_match else 0)
    completeness = (int(completeness_match.group(1)) if completeness_match else 0)
    fluency = (int(fluency_match.group(1)) if fluency_match else 0)

    return {
        "correctness": correctness,
        "completeness": completeness,
        "fluency": fluency
    }