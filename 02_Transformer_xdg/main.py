import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from transformer import ModelArgs, Transformer


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
IGNORE_INDEX = -1

DEFAULT_DATA_PATH = CURRENT_DIR / "notebooks" / "data.json"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "output"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: Optional[str] = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CharTokenizer:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.itos = list(tokens)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: Sequence[str]) -> "CharTokenizer":
        specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        chars = sorted(set("".join(texts)))
        vocab = specials + [ch for ch in chars if ch not in specials]
        return cls(vocab)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["tokens"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"tokens": self.itos}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        chars: List[str] = []
        for idx in token_ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if skip_special_tokens and token in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN}:
                continue
            chars.append(token)
        return "".join(chars)


def load_parallel_data(data_path: Path) -> Dict[str, Dict[str, List[str]]]:
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    data = json.loads(data_path.read_text(encoding="utf-8"))
    for split in ("train", "test"):
        if split not in data or "x" not in data[split] or "y" not in data[split]:
            raise ValueError(f"{data_path} 缺少 {split}.x / {split}.y 字段")
    return data


class TranslationDataset(Dataset):
    def __init__(
        self,
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: CharTokenizer,
        max_len: int,
    ) -> None:
        self.examples: List[Dict[str, object]] = []

        for source, target in zip(sources, targets):
            src_ids = [tokenizer.bos_id] + tokenizer.encode(source)[: max_len - 2] + [tokenizer.eos_id]
            tgt_ids = [tokenizer.bos_id] + tokenizer.encode(target)[: max_len - 2] + [tokenizer.eos_id]

            decoder_input_ids = tgt_ids[:-1]
            labels = tgt_ids[1:]

            self.examples.append(
                {
                    "source_text": source,
                    "target_text": target,
                    "source_ids": torch.tensor(src_ids, dtype=torch.long),
                    "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.examples[idx]


class TranslationCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        batch_size = len(batch)
        max_src_len = max(item["source_ids"].numel() for item in batch)  # type: ignore[index]
        max_tgt_len = max(item["decoder_input_ids"].numel() for item in batch)  # type: ignore[index]
        seq_len = max(max_src_len, max_tgt_len)

        source_ids = torch.full((batch_size, seq_len), self.pad_id, dtype=torch.long)
        decoder_input_ids = torch.full((batch_size, seq_len), self.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)

        source_texts: List[str] = []
        target_texts: List[str] = []

        for row, item in enumerate(batch):
            src = item["source_ids"]  # type: ignore[index]
            dec = item["decoder_input_ids"]  # type: ignore[index]
            lab = item["labels"]  # type: ignore[index]

            source_ids[row, : src.numel()] = src
            decoder_input_ids[row, : dec.numel()] = dec
            labels[row, : lab.numel()] = lab

            source_texts.append(item["source_text"])  # type: ignore[arg-type]
            target_texts.append(item["target_text"])  # type: ignore[arg-type]

        return {
            "source_ids": source_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "source_texts": source_texts,
            "target_texts": target_texts,
        }


class Seq2SeqTransformer(Transformer):
    def forward(  # type: ignore[override]
        self,
        source_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # transformer.py 中 cross-attention 的实现要求 encoder / decoder 序列长度一致，
        # 因此这里由 collate 保证 source_ids 和 decoder_input_ids 的长度都 pad 到同一个 T。
        src_emb = self.transformer.wte(source_ids)
        src_hidden = self.transformer.drop(self.transformer.wpe(src_emb))
        enc_out = self.transformer.encoder(src_hidden)

        dec_emb = self.transformer.wte(decoder_input_ids)
        dec_hidden = self.transformer.drop(self.transformer.wpe(dec_emb))
        dec_out = self.transformer.decoder(dec_hidden, enc_out)

        logits = self.lm_head(dec_out)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss


def build_model(model_args: ModelArgs, device: torch.device) -> Seq2SeqTransformer:
    if model_args.dim != model_args.n_embd:
        raise ValueError("当前 transformer.py 的位置编码/输出层要求 dim 与 n_embd 相同。")
    if model_args.dim % 2 != 0:
        raise ValueError("PositionalEncoding 要求 dim 为偶数。")
    if model_args.dim % model_args.n_heads != 0:
        raise ValueError("dim 必须能被 n_heads 整除。")
    return Seq2SeqTransformer(model_args).to(device)


def evaluate(model: Seq2SeqTransformer, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            source_ids = batch["source_ids"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(source_ids, decoder_input_ids, labels)
            if loss is None:
                continue
            total_loss += loss.item()
            total_steps += 1

    return total_loss / max(total_steps, 1)


def save_checkpoint(
    output_dir: Path,
    model: Seq2SeqTransformer,
    tokenizer: CharTokenizer,
    model_args: ModelArgs,
    history: Dict[str, List[float]],
    epoch: int,
    best_val_loss: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_args": asdict(model_args),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "history": history,
        },
        output_dir / "checkpoint.pt",
    )

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(
    output_dir: Path, device: torch.device
) -> Tuple[Seq2SeqTransformer, CharTokenizer, Dict[str, object]]:
    checkpoint_path = output_dir / "checkpoint.pt"
    tokenizer_path = output_dir / "tokenizer.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"找不到 tokenizer: {tokenizer_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    tokenizer = CharTokenizer.load(tokenizer_path)
    model_args = ModelArgs(**payload["model_args"])
    model = build_model(model_args, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, tokenizer, payload


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> int:
    logits = logits.clone()

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        cutoff = values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    probs = F.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_translation(
    model: Seq2SeqTransformer,
    tokenizer: CharTokenizer,
    source_text: str,
    device: torch.device,
    max_len: int,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> str:
    model.eval()

    source_ids = [tokenizer.bos_id] + tokenizer.encode(source_text)[: max_len - 2] + [tokenizer.eos_id]
    generated = [tokenizer.bos_id]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = max(len(source_ids), len(generated))
            if seq_len > max_len:
                break

            src_tensor = torch.full((1, seq_len), tokenizer.pad_id, dtype=torch.long, device=device)
            dec_tensor = torch.full((1, seq_len), tokenizer.pad_id, dtype=torch.long, device=device)

            src_tensor[0, : len(source_ids)] = torch.tensor(source_ids, dtype=torch.long, device=device)
            dec_tensor[0, : len(generated)] = torch.tensor(generated, dtype=torch.long, device=device)

            logits, _ = model(src_tensor, dec_tensor)
            next_token_logits = logits[0, len(generated) - 1]
            next_token_id = sample_next_token(next_token_logits, temperature=temperature, top_k=top_k)

            if next_token_id == tokenizer.eos_id:
                break
            generated.append(next_token_id)

            if len(generated) >= max_len:
                break

    return tokenizer.decode(generated[1:])


def preview_examples(
    model: Seq2SeqTransformer,
    tokenizer: CharTokenizer,
    dataset: Dict[str, Dict[str, List[str]]],
    device: torch.device,
    max_len: int,
    sample_count: int,
) -> None:
    sample_count = min(sample_count, len(dataset["test"]["x"]))
    if sample_count <= 0:
        return

    print("\n推理预览")
    print("=" * 80)
    for index in range(sample_count):
        source = dataset["test"]["x"][index]
        target = dataset["test"]["y"][index]
        prediction = generate_translation(
            model=model,
            tokenizer=tokenizer,
            source_text=source,
            device=device,
            max_len=max_len,
        )
        print(f"[样本 {index + 1}]")
        print(f"源句: {source}")
        print(f"参考: {target}")
        print(f"预测: {prediction}")
        print("-" * 80)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    data = load_parallel_data(data_path)
    if args.limit_train is not None:
        data["train"]["x"] = data["train"]["x"][: args.limit_train]
        data["train"]["y"] = data["train"]["y"][: args.limit_train]
    if args.limit_test is not None:
        data["test"]["x"] = data["test"]["x"][: args.limit_test]
        data["test"]["y"] = data["test"]["y"][: args.limit_test]

    tokenizer = CharTokenizer.build(
        data["train"]["x"] + data["train"]["y"] + data["test"]["x"] + data["test"]["y"]
    )

    train_dataset = TranslationDataset(
        data["train"]["x"], data["train"]["y"], tokenizer=tokenizer, max_len=args.max_len
    )
    test_dataset = TranslationDataset(
        data["test"]["x"], data["test"]["y"], tokenizer=tokenizer, max_len=args.max_len
    )

    collator = TranslationCollator(tokenizer.pad_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model_args = ModelArgs(
        n_embd=args.dim,
        n_heads=args.n_heads,
        dim=args.dim,
        dropout=args.dropout,
        max_len=args.max_len,
        vocab_size=tokenizer.vocab_size,
        n_layer=args.n_layer,
    )
    model = build_model(model_args, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"device: {device}")
    print(f"train size: {len(train_dataset)} | test size: {len(test_dataset)}")
    print(f"vocab size: {tokenizer.vocab_size}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            ncols=100,
            leave=False,
        )

        for batch in progress:
            source_ids = batch["source_ids"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(source_ids, decoder_input_ids, labels)
            if loss is None:
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        avg_val_loss = evaluate(model, test_loader, device)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                model_args=model_args,
                history=history,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )
            print(f"已保存最佳模型到: {output_dir}")

        if args.preview_samples > 0:
            preview_examples(
                model=model,
                tokenizer=tokenizer,
                dataset=data,
                device=device,
                max_len=args.max_len,
                sample_count=args.preview_samples,
            )


def predict(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    model, tokenizer, payload = load_checkpoint(output_dir, device)

    print(f"device: {device}")
    print(f"checkpoint epoch: {payload.get('epoch')}")
    print(f"best val loss: {payload.get('best_val_loss')}")

    texts = args.text or [
        "小冬瓜学大模型。",
        "注意力机制是 Transformer 的核心。",
        "大语言模型可以帮助我们提高效率。",
    ]

    for source in texts:
        prediction = generate_translation(
            model=model,
            tokenizer=tokenizer,
            source_text=source,
            device=device,
            max_len=model.args.max_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("=" * 80)
        print(f"源句: {source}")
        print(f"预测: {prediction}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer 训练与推理脚本")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=256)

    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--preview_samples", type=int, default=2)

    parser.add_argument("--text", action="append", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
