from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from preprocessing import SRLFrame, UtteranceSample, preprocess_single_utterance_for_politeness, simple_fallback_tokenize


class SRLDataset(Dataset):
    def __init__(self, samples: List[UtteranceSample], tokenizer, label2id: Dict[str, int], max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def _normalize_srl_label(self, lbl: str) -> str:
        return lbl.replace("R-", "").replace("C-", "")

    def _tokenize_utterance(self, words: List[str]):
        return self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    def _build_word_first_wp_fullidx(self, words: List[str]) -> List[int]:
        n_words = len(words)
        tmp = self.tokenizer(words, is_split_into_words=True, return_offsets_mapping=False)
        word_ids = tmp.word_ids()

        first_pos_by_wid = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in first_pos_by_wid:
                first_pos_by_wid[wid] = pos

        return [first_pos_by_wid[w] for w in range(n_words)]

    def _frame_to_tensors(self, words: List[str], frame: SRLFrame):
        n_words = len(words)
        norm_labels = [self._normalize_srl_label(lbl) for lbl in frame.labels]
        unk_id = self.label2id.get("O", 0)
        label_ids = [self.label2id.get(lbl, unk_id) for lbl in norm_labels]

        role_ids = [0] * n_words
        for i, tag in enumerate(norm_labels):
            if i == frame.predicate_word_idx:
                role_ids[i] = 1
            if "ARG0" in tag:
                role_ids[i] = 2
            elif "ARG1" in tag:
                role_ids[i] = 3
            elif "ARG2" in tag:
                role_ids[i] = 4
            elif "ARGM" in tag:
                role_ids[i] = 5

        arg0_mask = [1 if "ARG0" in tag else 0 for tag in norm_labels]
        arg1_mask = [1 if "ARG1" in tag else 0 for tag in norm_labels]
        arg2_mask = [1 if "ARG2" in tag else 0 for tag in norm_labels]
        argm_mask = [1 if "ARGM" in tag else 0 for tag in norm_labels]

        res = {
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "pred_word_idx": torch.tensor(frame.predicate_word_idx, dtype=torch.long),
            "role_ids": torch.tensor(role_ids, dtype=torch.long),
            "arg0_mask": torch.tensor(arg0_mask, dtype=torch.long),
            "arg1_mask": torch.tensor(arg1_mask, dtype=torch.long),
            "arg2_mask": torch.tensor(arg2_mask, dtype=torch.long),
            "argm_mask": torch.tensor(argm_mask, dtype=torch.long),
        }
        if frame.arg_head_idx is not None:
            res["arg_head_idx"] = torch.tensor(frame.arg_head_idx, dtype=torch.long)
        return res

    def __getitem__(self, idx):
        utt = self.samples[idx]
        words = utt.words

        enc = self._tokenize_utterance(words)
        sent_wp_ids = enc["input_ids"]

        input_ids = [self.tokenizer.cls_token_id] + sent_wp_ids + [self.tokenizer.sep_token_id]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            token_type_ids = token_type_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        word_first_wp_fullidx = self._build_word_first_wp_fullidx(words)
        frames = [self._frame_to_tensors(words, fr) for fr in utt.frames]

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "word_first_wp_fullidx": torch.tensor(word_first_wp_fullidx, dtype=torch.long),
            "sent_len": torch.tensor(len(words), dtype=torch.long),
            "frames": frames,
        }
        if utt.politeness is not None:
            item["politeness"] = torch.tensor(float(utt.politeness), dtype=torch.float)
        return item


def srl_collate_ulevel(batch: List[Dict], pad_token_id: int, pad_label_id: int = -100):
    B = len(batch)
    max_L = max(item["input_ids"].size(0) for item in batch)
    max_n = max(int(item["sent_len"]) for item in batch)
    max_F = max(len(item.get("frames", [])) for item in batch)

    input_ids = torch.full((B, max_L), pad_token_id, dtype=torch.long)
    token_type_ids = torch.zeros((B, max_L), dtype=torch.long)
    attention_mask = torch.zeros((B, max_L), dtype=torch.long)

    word_first_wp_fullidx = torch.full((B, max_n), -1, dtype=torch.long)
    sent_lens = torch.zeros((B,), dtype=torch.long)
    sentence_mask = torch.zeros((B, max_n), dtype=torch.bool)

    frames_mask = torch.zeros((B, max_F), dtype=torch.bool)
    frames_labels = torch.full((B, max_F, max_n), pad_label_id, dtype=torch.long)
    frames_pred_word_idx = torch.zeros((B, max_F), dtype=torch.long)
    frames_role_ids = torch.zeros((B, max_F, max_n), dtype=torch.long)
    frames_arg0_mask = torch.zeros((B, max_F, max_n), dtype=torch.long)
    frames_arg1_mask = torch.zeros((B, max_F, max_n), dtype=torch.long)
    frames_arg2_mask = torch.zeros((B, max_F, max_n), dtype=torch.long)
    frames_argm_mask = torch.zeros((B, max_F, max_n), dtype=torch.long)

    has_politeness = "politeness" in batch[0]
    if has_politeness:
        politeness = torch.zeros((B,), dtype=torch.float)

    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        n = int(item["sent_len"])

        input_ids[i, :L] = item["input_ids"]
        token_type_ids[i, :L] = item["token_type_ids"]
        attention_mask[i, :L] = item["attention_mask"]

        word_first_wp_fullidx[i, :n] = item["word_first_wp_fullidx"]
        sent_lens[i] = n
        sentence_mask[i, :n] = True

        if has_politeness:
            politeness[i] = item["politeness"]

        for f, fr in enumerate(item.get("frames", [])):
            frames_mask[i, f] = True
            frames_labels[i, f, :n] = fr["labels"]
            frames_pred_word_idx[i, f] = fr["pred_word_idx"]
            frames_role_ids[i, f, :n] = fr["role_ids"]
            frames_arg0_mask[i, f, :n] = fr["arg0_mask"]
            frames_arg1_mask[i, f, :n] = fr["arg1_mask"]
            frames_arg2_mask[i, f, :n] = fr["arg2_mask"]
            frames_argm_mask[i, f, :n] = fr["argm_mask"]

    res = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "word_first_wp_fullidx": word_first_wp_fullidx,
        "sentence_mask": sentence_mask,
        "sent_lens": sent_lens,
        "frames_mask": frames_mask,
        "frames_labels": frames_labels,
        "frames_pred_word_idx": frames_pred_word_idx,
        "frames_role_ids": frames_role_ids,
        "frames_arg0_mask": frames_arg0_mask,
        "frames_arg1_mask": frames_arg1_mask,
        "frames_arg2_mask": frames_arg2_mask,
        "frames_argm_mask": frames_argm_mask,
    }
    if has_politeness:
        res["politeness"] = politeness
    return res


class DirectionalSRL(nn.Module):
    def __init__(
        self,
        bert_name: str,
        num_labels: int,
        use_indicator: bool = True,
        indicator_dim: int = 10,
        lstm_hidden: int = 768,
        mlp_hidden: int = 300,
        dropout: float = 0.1,
        use_distance: bool = True,
        pos_dim: int = 50,
        max_distance: int = 128,
        num_roles: int = 6,
        role_dim: int = 32,
        attn_dim: int = 256,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(bert_name)
        self.bert = AutoModel.from_pretrained(bert_name)
        self.use_indicator = use_indicator
        self.use_distance = use_distance
        self.max_distance = max_distance

        bert_dim = self.config.hidden_size
        in_dim = bert_dim + (indicator_dim if use_indicator else 0)

        if use_indicator:
            self.indicator_emb = nn.Embedding(num_roles, indicator_dim)
        if use_distance:
            self.pos_emb = nn.Embedding(2 * max_distance + 1, pos_dim)
            in_dim += pos_dim

        self.bilstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(lstm_hidden, attn_dim)
        self.W_k = nn.Linear(lstm_hidden, attn_dim)
        self.W_v = nn.Linear(lstm_hidden, attn_dim)
        self.attn_layer_norm = nn.LayerNorm(attn_dim)
        self.politeness_head = nn.Sequential(
            nn.Linear(lstm_hidden + attn_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        word_first_wp_fullidx,
        sentence_mask,
        sent_lens,
        frames_mask,
        frames_pred_word_idx,
        frames_role_ids=None,
        frames_arg0_mask=None,
        frames_arg2_mask=None,
        frames_argm_mask=None,
        politeness=None,
        **kwargs,
    ):
        B, _ = input_ids.size()
        device = input_ids.device

        bert_out = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        H = bert_out.last_hidden_state

        gather_idx = word_first_wp_fullidx.clone()
        gather_idx[gather_idx < 0] = 0
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, H.size(-1))

        H_words = torch.gather(H, dim=1, index=gather_idx)
        H_words = H_words * sentence_mask.unsqueeze(-1)

        F = frames_mask.size(1)
        BF = B * F
        N = H_words.size(1)

        Hf = H_words.unsqueeze(1).expand(B, F, N, H_words.size(-1)).contiguous().view(BF, N, H_words.size(-1))
        sent_mask_f = sentence_mask.unsqueeze(1).expand(B, F, N).contiguous().view(BF, N)
        lengths_f = sent_lens.unsqueeze(1).expand(B, F).contiguous().view(BF)
        pred_idx_f = frames_pred_word_idx.contiguous().view(BF)

        X = Hf
        if self.use_indicator and frames_role_ids is not None:
            role_ids_f = frames_role_ids.clamp(0, 5).contiguous().view(BF, N)
            X = torch.cat([X, self.indicator_emb(role_ids_f)], dim=-1)

        if self.use_distance:
            positions = torch.arange(N, device=device).unsqueeze(0).expand(BF, -1)
            rel = (positions - pred_idx_f.unsqueeze(1)).clamp(-self.max_distance, self.max_distance) + self.max_distance
            X = torch.cat([X, self.pos_emb(rel)], dim=-1)

        valid_frame_mask = frames_mask.reshape(-1).bool()
        X_valid = X[valid_frame_mask]
        lengths_valid = lengths_f[valid_frame_mask].clamp(min=1)

        if X_valid.size(0) == 0:
            score = torch.zeros((B,), device=device)
            loss = torch.tensor(0.0, device=device)
            attn_weights = torch.zeros((B, F, N), device=device)
            if politeness is not None:
                loss = nn.MSELoss()(score, politeness)
            return score, loss, attn_weights

        packed = pack_padded_sequence(X_valid, lengths=lengths_valid.detach().cpu(), batch_first=True, enforce_sorted=False)
        G_packed, _ = self.bilstm(packed)
        G_valid, _ = pad_packed_sequence(G_packed, batch_first=True, total_length=N)
        G_valid = self.dropout(G_valid)

        Hdim = G_valid.size(-1)
        G = torch.zeros((X.size(0), N, Hdim), device=G_valid.device, dtype=G_valid.dtype)
        G[valid_frame_mask] = G_valid

        arg0_f = frames_arg0_mask.contiguous().view(BF, N) if frames_arg0_mask is not None else None
        arg2_f = frames_arg2_mask.contiguous().view(BF, N) if frames_arg2_mask is not None else None
        argm_f = frames_argm_mask.contiguous().view(BF, N) if frames_argm_mask is not None else None

        batch_idx = torch.arange(BF, device=device)
        gp = G[batch_idx, pred_idx_f.clamp(min=0, max=N - 1), :]

        if arg0_f is not None:
            arg0_sum = arg0_f.sum(dim=1, keepdim=True)
            has_arg0 = (arg0_sum > 0).float()
            denom = arg0_sum.clamp(min=1.0)
            arg0_vec = (G * arg0_f.unsqueeze(-1)).sum(dim=1) / denom
            query_source = has_arg0 * arg0_vec + (1 - has_arg0) * gp
        else:
            query_source = gp

        if arg2_f is not None and argm_f is not None:
            target_mask = (arg2_f + argm_f).clamp(max=1)
            target_mask = target_mask * sent_mask_f
        else:
            target_mask = sent_mask_f

        Q = self.W_q(query_source).unsqueeze(1)
        K = self.W_k(G)
        V = self.W_v(G)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)
        attn_scores = attn_scores.masked_fill(target_mask.unsqueeze(1) == 0, -10000.0)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, V).squeeze(1)
        context = self.attn_layer_norm(context)

        features = torch.cat([query_source, context], dim=1)
        score_frame = self.politeness_head(features).squeeze(-1)

        score_frame = score_frame.view(B, F)
        attn_weights = attn_weights.squeeze(1).view(B, F, N)

        fm = frames_mask.float()
        denom = fm.sum(dim=1).clamp(min=1.0)
        score_utt = (score_frame * fm).sum(dim=1) / denom
        attn_weights = attn_weights * fm.unsqueeze(-1)

        loss = torch.tensor(0.0, device=device)
        if politeness is not None:
            loss = nn.MSELoss()(score_utt, politeness)
        return score_utt, loss, attn_weights


def load_directional_model(ckpt_file, device):
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    label2id = checkpoint["label2id"]

    model = DirectionalSRL(
        bert_name="bert-base-cased",
        num_labels=len(label2id),
        use_indicator=True,
        use_distance=True,
        indicator_dim=10,
        lstm_hidden=768,
        mlp_hidden=300,
        pos_dim=50,
        max_distance=128,
        dropout=0.0,
    )

    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model, label2id


def predict_politeness(
    utterance: str,
    ckpt_file: str,
    device: Optional[str] = None,
    max_length: int = 256,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model, label2id = load_directional_model(ckpt_file, device)
    info, sample = preprocess_single_utterance_for_politeness(utterance)

    if len(sample.words) == 0:
        fallback_words = simple_fallback_tokenize(utterance.strip())
        if len(fallback_words) == 0:
            raise ValueError("Utterance became empty after preprocessing.")
        sample.words = fallback_words
        sample.frames = []
        if len(info.get("srl_descriptions", [])) == 0:
            info["srl_descriptions"] = [
                f"No SRL frame detected. Fallback tokenization used: {' '.join(fallback_words)}"
            ]

    if len(sample.frames) == 0:
        sample.frames = [
            SRLFrame(
                predicate_word_idx=0,
                labels=["O"] * len(sample.words),
                predicate_form=sample.words[0] if len(sample.words) > 0 else None,
                arg_head_idx=None,
            )
        ]
        if "srl_descriptions" not in info or info["srl_descriptions"] is None:
            info["srl_descriptions"] = []

    ds = SRLDataset([sample], tokenizer, label2id, max_length=max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: srl_collate_ulevel(b, pad_token_id=pad_id, pad_label_id=-100),
    )

    with torch.no_grad():
        batch = next(iter(dl))
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        preds, loss, attn = model(**batch)

    return {
        "utterance": utterance,
        "tokenized_words": sample.words,
        "num_frames": len(sample.frames),
        "predicted_politeness": float(preds[0].detach().cpu().item()),
        "attention_weights": attn[0].detach().cpu() if attn is not None else None,
        "srl_sample": sample,
        "srl_descriptions": info.get("srl_descriptions", []),
    }
