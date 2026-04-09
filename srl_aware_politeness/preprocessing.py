import re
import unicodedata
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import spacy
from huggingface_hub import hf_hub_download, snapshot_download

_nlp = None
_srl_ready = False


def get_nlp(model_name: str = "en_core_web_md"):
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model_name)
    return _nlp


def initialize_srl(
    repo_id: str = "yeomtong/srl_bert_model",
    filename: str = "best_srl_fr_Oct_23_Ver2.ckpt",
    bert_name: str = "bert-base-cased",
):
    global _srl_ready
    if _srl_ready:
        return

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    repo_dir = snapshot_download(repo_id)
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    from predictor import srl_init

    srl_init(ckpt_path, bert_name=bert_name)
    _srl_ready = True


@dataclass
class SRLFrame:
    predicate_word_idx: int
    labels: List[str]
    predicate_form: Optional[str] = None
    arg_head_idx: Optional[List[int]] = None


@dataclass
class UtteranceSample:
    words: List[str]
    frames: List[SRLFrame]
    politeness: Optional[float] = None


def clean_text_for_srl(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = text.replace("Â", "").replace("Ã", "A").replace("â€”", "—").replace("â€“", "-")
    text = text.replace("±", "+/-").replace("°", " degrees ")
    text = re.sub(r"\([^)]{40,}\)", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text.split()) > 150:
        text = " ".join(text.split()[:150])

    return text


def simple_fallback_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def extract_arg_heads(words: List[str], tags: List[str]) -> List[int]:
    pred_idx = -1
    arg0_idx = -1
    arg1_idx = -1
    arg2_idx = -1
    argm_idx = -1

    for i, tag in enumerate(tags):
        if tag == "B-V" and pred_idx == -1:
            pred_idx = i
        if tag.startswith("B-ARG0") and arg0_idx == -1:
            arg0_idx = i
        if tag.startswith("B-ARG1") and arg1_idx == -1:
            arg1_idx = i
        if tag.startswith("B-ARG2") and arg2_idx == -1:
            arg2_idx = i
        if tag.startswith("B-ARGM") and argm_idx == -1:
            argm_idx = i

    return [pred_idx, arg0_idx, arg1_idx, arg2_idx, argm_idx]


def srl_info_for_single_utterance(utterance_text: str, politeness_score=None) -> Dict:
    initialize_srl()
    from visualizer import prediction_formatted

    utterance_text = clean_text_for_srl(utterance_text)
    if not isinstance(utterance_text, str) or not utterance_text.strip():
        return {"srl_frames": [], "srl_descriptions": [], "politeness": politeness_score}

    nlp = get_nlp()
    doc = nlp(utterance_text)
    sentences = [sent.text.strip() for sent in doc.sents]

    record = {"srl_frames": [], "srl_descriptions": [], "politeness": politeness_score}

    for sent in sentences:
        if not sent:
            continue

        srl_out = prediction_formatted(sent)
        words = srl_out.get("words", [])
        verbs = srl_out.get("verbs", [])

        for frame in verbs:
            tags = frame.get("tags", [])
            desc = frame.get("description", "")
            verb = frame.get("verb", None)

            if desc:
                record["srl_descriptions"].append(desc)

            if len(tags) != len(words):
                continue

            heads = extract_arg_heads(words, tags)
            pred_idx = heads[0]
            if pred_idx < 0:
                continue

            record["srl_frames"].append(
                {
                    "words": words,
                    "predicate_word_idx": pred_idx,
                    "labels": tags,
                    "arg_head_idx": heads,
                    "predicate_form": verb if verb is not None else (words[pred_idx] if 0 <= pred_idx < len(words) else None),
                    "description": desc,
                }
            )

    return record


def convert_example_to_utterance_level(ex: Dict, sep_token: Optional[str] = None) -> UtteranceSample:
    srl_frames = ex.get("srl_frames", [])
    politeness = ex.get("politeness", None)

    if not srl_frames:
        return UtteranceSample(words=[], frames=[], politeness=politeness)

    sentences: List[List[str]] = []
    sent_to_id: Dict[Tuple[str, ...], int] = {}

    for fr in srl_frames:
        w = tuple(fr["words"])
        if w not in sent_to_id:
            sent_to_id[w] = len(sentences)
            sentences.append(list(fr["words"]))

    utter_words: List[str] = []
    offsets: List[int] = []

    for si, sent_words in enumerate(sentences):
        offsets.append(len(utter_words))
        utter_words.extend(sent_words)
        if sep_token is not None and si != len(sentences) - 1:
            utter_words.append(sep_token)

    aligned_frames: List[SRLFrame] = []

    for fr in srl_frames:
        sent_words = fr["words"]
        sid = sent_to_id[tuple(sent_words)]
        start = offsets[sid]

        pred_idx_utt = start + fr["predicate_word_idx"]

        labels_utt = ["O"] * len(utter_words)
        for j, lab in enumerate(fr["labels"]):
            labels_utt[start + j] = lab

        arg_head = fr.get("arg_head_idx", None)
        if arg_head is not None:
            shifted = []
            for x in arg_head:
                shifted.append(x + start if isinstance(x, int) and x >= 0 else x)
            arg_head = shifted

        aligned_frames.append(
            SRLFrame(
                predicate_word_idx=pred_idx_utt,
                labels=labels_utt,
                predicate_form=fr.get("predicate_form", None),
                arg_head_idx=arg_head,
            )
        )

    return UtteranceSample(words=utter_words, frames=aligned_frames, politeness=politeness)


def preprocess_single_utterance_for_politeness(utterance: str):
    sent_level = srl_info_for_single_utterance(utterance, politeness_score=None)
    utt_level = convert_example_to_utterance_level(sent_level)
    return sent_level, utt_level
