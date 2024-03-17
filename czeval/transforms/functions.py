import re


def extract_letter(answer: str, letters: list[str] = ["A", "B", "C"], regexes: list[str] = []) -> str:
    """
    Extracts letter from answer from the end
    """

    # First try regexes
    for regex in regexes:
        match = re.search(regex, answer)
        if match is not None:
            return match.group("letter")


    answer_regex = re.compile(rf"({'|'.join(letters)})")
    # Find the first lette from the end
    match = answer_regex.search(answer[::-1])
    if match is None:
        return "NA"

    # Extract letter
    letter = match.group(1)
    return letter
