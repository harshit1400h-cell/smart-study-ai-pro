def calculate_quiz_score(questions: list[dict], answers: dict[int, str]) -> tuple[int, float]:
    if not questions:
        return 0, 0.0
    correct = sum(
        answers.get(index) == question["answer"]
        for index, question in enumerate(questions)
    )
    return correct, correct / len(questions) * 100


def display_source(source: dict) -> str:
    page = f", page {source['page']}" if source.get("page") else ""
    return f"{source['filename']}{page}"
