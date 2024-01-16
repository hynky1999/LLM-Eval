import re


def extract_letter(answer: str, letters: list[str] = ["A", "B", "C"]) -> str:
    answer_regex = re.compile(rf"({'|'.join(letters)})")
    """
    Extracts letter from answer from the end
    """
    # Find the first lette from the end
    match = answer_regex.search(answer[::-1])
    if match is None:
        return "NA"

    # Extract letter
    letter = match.group(1)
    return letter
