import json
import re
from typing import Dict

from openai import OpenAI


def create_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def politeness_band(score: float) -> str:
    if 0.0 <= score < 0.8:
        return "Rude"
    elif 0.8 <= score < 1.6:
        return "Slightly Rude"
    elif 1.6 <= score < 2.4:
        return "Neutral"
    elif 2.4 <= score < 3.2:
        return "Slightly Polite"
    elif 3.2 <= score <= 4.0:
        return "Polite"
    return "Out of range"


def format_srl_descriptions_for_prompt(srl_descriptions):
    if not srl_descriptions:
        return "No semantic role descriptions found."
    return "\n".join(f"Frame {i}: {desc}" for i, desc in enumerate(srl_descriptions, 1))


def build_explanation_prompt_from_result(result: Dict) -> str:
    utterance = result["utterance"]
    score = result["predicted_politeness"]
    band = politeness_band(score)
    srl_block = format_srl_descriptions_for_prompt(result["srl_descriptions"])

    return f"""
You are generating a concise post-hoc explanation for the output of an SRL-aware politeness model.

Model setting:
- utterance-level directional SRL politeness model
- attention direction: ARG0 -> ARG2 + ARGM
- score range: 0 to 4

Politeness bands:
- Rude: 0.0 - 0.8
- Slightly Rude: 0.8 - 1.6
- Neutral: 1.6 - 2.4
- Slightly Polite: 2.4 - 3.2
- Polite: 3.2 - 4.0

Important instructions:
- This is a post-hoc explanation, not a claim about the model's exact internal reasoning.
- Explain the prediction using linguistic features only.
- Focus especially on semantic roles, predicates, arguments, and modifiers.
- Explicitly refer to roles such as ARG0, ARG2, and ARGM when relevant.
- If ARG2 is absent, note that the interpretation relies mainly on ARGM.
- Do not make personal, moral, or social judgments about the speaker.
- Do not say "the speaker is rude" or "the speaker is polite."
- Instead, describe how the utterance is linguistically framed and how that framing may align with the predicted score.
- Use cautious language such as "suggests," "is framed as," "may be interpreted as," or "likely reflects."
- Keep both fields concise.
- The combined total length of "frame_summary" and "explanation" must not exceed 1024 characters.
- Return valid JSON only, with exactly two keys: "frame_summary" and "explanation".

Task:
Return a JSON object with exactly these two fields:
1. "frame_summary"
2. "explanation"

Utterance:
{utterance}

Politeness score:
{score:.3f} / 4.0

Band:
{band}

Semantic role descriptions:
{srl_block}

Output valid JSON only. No markdown.
""".strip()


def explain_with_gpt_oss_from_result(client: OpenAI, result: Dict, model_name: str = "gpt-oss-120b"):
    prompt = build_explanation_prompt_from_result(result)

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": (
                    "You produce concise post-hoc explanations in JSON. "
                    'Return exactly one JSON object with keys "frame_summary" and "explanation". '
                    "Use linguistic features and semantic roles only. "
                    "Do not make personal or moral judgments about the speaker."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = {"frame_summary": "", "explanation": text}
        else:
            parsed = {"frame_summary": "", "explanation": text}

    if "frame_summary" not in parsed:
        parsed["frame_summary"] = ""
    if "explanation" not in parsed:
        parsed["explanation"] = ""

    return parsed
