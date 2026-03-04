def get_prompt(method, query, top_passages_str, **kwargs):
    user_prompts = {
        "thinkqe": f"""Given a question "{query}" and its possible answering passages:
{top_passages_str}

Follow these steps:
1) CONCEPTS (one short line): list 1–2 core concepts, each 1–3 words, separated by commas.
2) PASSAGE (one paragraph, <= 80 words): write a concise answer that directly answers the question.
- Use information from passages only when it clearly supports the concepts; if you use a passage, append "[P#]" where # is the passage index.
- Avoid speculation and unrelated details.
Return ONLY the two items in order (CONCEPTS line then PASSAGE paragraph).""",
    }
    return user_prompts[method]



