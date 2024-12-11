QUESTION_PLACEHOLDER = '[INSERT PROMPT HERE]'


def synthesis_message(question, prompt):
    if QUESTION_PLACEHOLDER not in prompt:
        return None

    return prompt.replace(QUESTION_PLACEHOLDER, question)
