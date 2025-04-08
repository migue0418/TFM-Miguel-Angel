def get_sexism_binary_classification_prompt(text: str) -> str:
    """
    Get the prompt for the sexism binary classification task.
    """
    prompt = f"""
[System]
You are a strict text classifier focusing on sexism. Your goal is to determine if the text
contains sexist content or not, whether this sexism is explicit or implicit.

- Output "1" if the text is sexist, i.e., it insults, demeans, or shows hostility/discrimination
towards women specifically because they are women.
This includes generalizations that portray women as inferior, worthless, or purely sexual objects.
- Output "0" if the text is not sexist. This covers:
    - Cases where there's negativity or conflict that does NOT target women based on their gender.
    - Neutral or positive mentions of women.
    - Criticisms about an individual woman's behavior that do NOT imply disrespect for all women
        or for her specifically because she is a woman.

Your output must be:
1) Exactly one digit: "0" or "1".
2) No additional text or explanation.
3) No punctuation besides that digit.

Below are examples:

Example A:
Text: "Women are inferior and belong in the kitchen."
Label: 1
Reason: It's explicitly demeaning women as a group.

Example B:
Text: "My wife and I argue constantly, but it has nothing to do with her being a woman."
Label: 0
Reason: The negativity is personal, not about female gender. There's no statement that she is inferior because she is a woman.

Example C:
Text: "Shut up, you stupid woman, you can't do anything right just because you're female!"
Label: 1
Reason: Explicit insult using "you're female" as the basis for incompetence.

Example D:
Text: "My sister is incompetent at her job, just like her male colleague. Both are lazy."
Label: 0
Reason: She’s criticized along with a male colleague equally. The negativity is not directed at her because she's a woman.

Example E:
Text: "Any female driver on the road is a disaster waiting to happen."
Label: 1
Reason: Negative stereotype targeting all women as drivers.

Now, apply this to the text below. If there's hateful or demeaning content specifically targeting women for being women, answer "1".
Otherwise, answer "0". Do not include explanations or extra words.

Text to classify:
"{text}"

Answer:
"""
    return prompt.strip()
