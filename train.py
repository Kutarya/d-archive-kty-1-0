import argparse
import ast
import gc
import hashlib
import io
import json
import math
import random
import re
import signal
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

import config
from model import KTYModel, load_model_checkpoint

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / config.CHECKPOINT_NAME
NEW_DATASETS = ("sertlik.jsonl", "sohbet.jsonl", "matematik.jsonl", "python.jsonl", "reasoning.jsonl")
ANCHORS = {"selam", "merhaba", "naber", "nasılsın", "sen kimsin", "adın ne",
           "kimsin sen", "ismin ne", "günaydın", "iyi akşamlar", "iyi geceler", "2+2"}


def read_records(path):
    records = []
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name}:{line_number}: geçersiz JSON") from error
            if not isinstance(record, dict) or set(record) != {"user", "assistant"}:
                raise ValueError(f"{path.name}:{line_number}: user/assistant şeması gerekli")
            if any(not isinstance(v, str) or not v.strip() for v in record.values()):
                raise ValueError(f"{path.name}:{line_number}: boş veya metin olmayan alan")
            for value in record.values():
                value.encode("utf-8", errors="strict")
                if "\ufffd" in value:
                    raise ValueError(f"{path.name}:{line_number}: bozuk UTF-8 karakteri")
            records.append(record)
    return records


def normalized_user(text):
    text = text.casefold().strip()
    text = text.removeprefix(config.REASONING_TAG.casefold()).strip()
    text = re.sub(r"\d+(?:[.,]\d+)?", "#", text)
    return re.sub(r"\s+", " ", text).rstrip(".?!")


def is_validation(user):
    if user.strip().casefold().rstrip(".?!") in ANCHORS:
        return False
    group = normalized_user(user)
    value = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "big")
    return value / 2**64 < config.VALIDATION_FRACTION


def encode_chat(record):
    prompt = f"Kullanıcı: {record['user'].strip()}\nKTY: "
    answer = record["assistant"].strip() + f"\n{config.END_MARKER}\n"
    prefix = prompt.encode("utf-8")
    response = answer.encode("utf-8")
    return prefix + response, bytes(len(prefix)) + bytes([1]) * len(response)


def validate_datasets(records_by_name):
    for name, records in records_by_name.items():
        keys = [(r["user"], r["assistant"]) for r in records]
        duplicates = len(keys) - len(set(keys))
        if name in NEW_DATASETS and (len(records) < 2000 or duplicates):
            raise ValueError(f"{name}: >=2000 satır ve sıfır tam tekrar gerekli")
        code_count = 0
        if name in ("python.jsonl", "reasoning.jsonl"):
            for i, record in enumerate(records, 1):
                answer_blocks = re.findall(r"```python\s*\n(.*?)```", record["assistant"], re.S)
                blocks = answer_blocks
                if name == "reasoning.jsonl":
                    blocks += re.findall(r"```python\s*\n(.*?)```", record["user"], re.S)
                    if not (record["user"].startswith(config.REASONING_TAG)
                            and record["assistant"].startswith("Çözüm: ")
                            and "\nCevap: " in record["assistant"]):
                        raise ValueError(f"{name}:{i}: reasoning biçimi geçersiz")
                elif not blocks:
                    raise ValueError(f"{name}:{i}: Python kodu eksik")
                for code in blocks:
                    ast.parse(code, filename=f"{name}:{i}")
                    compile(code, f"{name}:{i}", "exec")
                    code_count += 1
        counts = Counter(r["assistant"] for r in records)
        print(f"{name}: {len(records)} satır; JSON/şema/UTF-8 OK; "
              f"tam tekrar={duplicates}; farklı cevap={len(counts)}; "
              f"en sık cevap={max(counts.values(), default=0)}"
              + (f"; Python syntax={code_count} OK" if code_count else ""), flush=True)


class TrainingData:
    def __init__(self):
        self.pools = {"train": {}, "validation": {}}
        self.anchors = {}
        self.stats = {}
        self.rng = random.Random(config.SEED)
        self.fingerprint = hashlib.sha256()
        records_by_name = {}
        for name in ("kty_chat.jsonl",) + NEW_DATASETS:
            path = BASE_DIR / "data" / name
            if name == "kty_chat.jsonl" and not path.exists():
                continue
            records_by_name[name] = read_records(path)
        validate_datasets(records_by_name)
        seen = set()
        groups = {"train": set(), "validation": set()}
        for name, records in records_by_name.items():
            self._fingerprint_file(name)
            train, validation, anchors = [], [], []
            skipped = 0
            for record in records:
                key = (record["user"].strip().casefold(), record["assistant"].strip())
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)
                item = encode_chat(record)
                val = is_validation(record["user"])
                split = "validation" if val else "train"
                groups[split].add(normalized_user(record["user"]))
                (validation if val else train).append(item)
                if not val and record["user"].strip().casefold().rstrip(".?!") in ANCHORS:
                    anchors.append(item)
            self.pools["train"][name] = train
            self.pools["validation"][name] = validation
            self.anchors[name] = anchors
            self.stats[name] = {
                "records": len(records), "train": len(train), "validation": len(validation),
                "skipped_duplicates": skipped, "bytes": sum(len(t[0]) for t in train + validation),
            }
        if groups["train"] & groups["validation"]:
            raise ValueError("Normalize edilmiş soru grubu train/validation arasında sızdı.")
        name = "train.txt"
        path = BASE_DIR / "data" / name
        text = path.read_text(encoding="utf-8", errors="strict")
        self._fingerprint_file(name)
        train, validation = [], []
        for line in text.splitlines():
            if not line.strip():
                continue
            raw = (line + "\n").encode("utf-8")
            item = raw, bytes([1]) * len(raw)
            (validation if is_validation(line) else train).append(item)
        self.pools["train"][name] = train
        self.pools["validation"][name] = validation
        self.anchors[name] = []
        self.stats[name] = {
            "records": len(text.splitlines()), "train": len(train), "validation": len(validation),
            "skipped_duplicates": 0, "bytes": len(text.encode("utf-8")),
        }
        for split, pools in self.pools.items():
            if not any(pools.values()):
                raise ValueError(f"{split} verisi boş.")
            for name, pool in pools.items():
                if not pool and split == "train":
                    raise ValueError(f"{name}: train split boş.")
        self.fingerprint = self.fingerprint.hexdigest()

    def _fingerprint_file(self, name):
        self.fingerprint.update(name.encode())
        self.fingerprint.update((BASE_DIR / "data" / name).read_bytes())

    @property
    def numel(self):
        return sum(s["bytes"] for s in self.stats.values())


def load_training_data():
    data = TrainingData()
    return data, data.stats


def make_batch(data, context_size, batch_size, device, split="train"):
    pools = data.pools[split]
    names = [name for name, pool in pools.items() if pool]
    source_weights = [config.DATASET_WEIGHTS[name] for name in names]
    xs, ys, weights = [], [], []
    for _ in range(batch_size):
        name = data.rng.choices(names, weights=source_weights, k=1)[0]
        pool = pools[name]
        anchors = data.anchors[name]
        raw, mask = bytearray(), bytearray()
        while len(raw) < context_size + 1:
            use_anchor = split == "train" and anchors and data.rng.random() < 0.25
            sample_pool = anchors if use_anchor else pool
            text, labels = sample_pool[data.rng.randrange(len(sample_pool))]
            offset = 0
            if not raw and len(text) > context_size + 1 and data.rng.random() < 0.5:
                offset = data.rng.randrange(len(text) - context_size)
            raw.extend(text[offset:])
            mask.extend(labels[offset:])
        tokens = torch.frombuffer(raw, dtype=torch.uint8).long()
        labels = torch.frombuffer(mask, dtype=torch.uint8).float()
        xs.append(tokens[:context_size])
        ys.append(tokens[1:context_size + 1])
        weights.append(config.USER_LOSS_WEIGHT + (1 - config.USER_LOSS_WEIGHT)
                       * labels[1:context_size + 1])
    return (torch.stack(xs).to(device), torch.stack(ys).to(device),
            torch.stack(weights).to(device))


def sequence_loss(logits, targets, weights=None):
    losses = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1),
                             reduction="none").reshape_as(targets)
    return losses.mean() if weights is None else (losses * weights).sum() / weights.sum()


def learning_rate(step):
    if step <= config.WARMUP_STEPS:
        return config.LEARNING_RATE * step / max(1, config.WARMUP_STEPS)
    progress = min(1.0, (step - config.WARMUP_STEPS) /
                   max(1, config.TRAIN_STEPS - config.WARMUP_STEPS))
    return config.MIN_LEARNING_RATE + (config.LEARNING_RATE - config.MIN_LEARNING_RATE) * (
        1 + math.cos(math.pi * progress)) / 2


def checkpoint_payload(model, optimizer, step, data=None, metrics=None):
    return {
        "model_name": "KTY-1.0", "checkpoint_version": 4,
        "chat_format": config.CHAT_FORMAT, "end_marker": config.END_MARKER,
        "reasoning_tag": config.REASONING_TAG,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "step": step, "config": model.config,
        "training_config": {k: getattr(config, k) for k in (
            "LEARNING_RATE", "MIN_LEARNING_RATE", "TRAIN_STEPS", "BATCH_SIZE",
            "GRAD_ACCUM_STEPS", "WARMUP_STEPS", "CPU_THREADS", "SEED", "USER_LOSS_WEIGHT",
            "VALIDATION_FRACTION", "WEIGHT_DECAY", "DATASET_WEIGHTS")},
        "torch_rng_state": torch.get_rng_state(),
        "data_rng_state": data.rng.getstate() if data else None,
        "dataset_fingerprint": data.fingerprint if data else None,
        "metrics": metrics or {},
    }


def output_path(path):
    path = Path(path).resolve()
    if path.parent != (BASE_DIR / "checkpoints").resolve() or path.suffix != ".pt":
        raise ValueError("Checkpoint yalnızca workspace/checkpoints/*.pt olabilir.")
    return path


def save_checkpoint(path, model, optimizer, step, data=None, metrics=None):
    path = output_path(path)
    try:
        stream = path.open("xb")
    except FileExistsError as error:
        raise FileExistsError(f"Checkpoint korunuyor, üzerine yazılmadı: {path}") from error
    try:
        with stream:
            torch.save(checkpoint_payload(model, optimizer, step, data, metrics), stream)
    except BaseException:
        # Yalnız bu çağrının oluşturduğu eksik dosya kaldırılır.
        path.unlink(missing_ok=True)
        raise


def evaluate(model, data):
    state = data.rng.getstate()
    was_training = model.training
    data.rng.seed(config.SEED + 101)
    model.eval()
    weighted, plain = [], []
    try:
        with torch.inference_mode():
            for _ in range(config.EVAL_BATCHES):
                x, y, weights = make_batch(data, model.context_size, config.BATCH_SIZE,
                                          "cpu", "validation")
                logits = model(x)
                weighted.append(sequence_loss(logits, y, weights).item())
                plain.append(sequence_loss(logits, y).item())
    finally:
        data.rng.setstate(state)
        model.train(was_training)
    return {"validation_loss": sum(plain) / len(plain),
            "validation_weighted_loss": sum(weighted) / len(weighted)}


def smoke_test(data):
    torch.manual_seed(config.SEED)
    saved_rng = data.rng.getstate()
    model = KTYModel().cpu().eval()
    x, y, weights = make_batch(data, config.CONTEXT_SIZE, config.BATCH_SIZE, "cpu")
    assert x.shape == y.shape == weights.shape == (config.BATCH_SIZE, config.CONTEXT_SIZE)
    assert torch.equal(y[:, :-1], x[:, 1:])
    started = time.perf_counter()
    logits = model(x)
    assert logits.shape == (*x.shape, 256) and torch.isfinite(logits).all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss = sequence_loss(logits, y, weights)
    assert torch.isfinite(loss)
    plain = sequence_loss(logits, y)
    manual = -logits.log_softmax(-1).gather(-1, y.unsqueeze(-1)).mean()
    torch.testing.assert_close(plain, manual)
    before = model.lm_head.weight.detach().clone()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
    optimizer.step()
    elapsed = time.perf_counter() - started
    assert not torch.equal(before, model.lm_head.weight)
    optimizer.zero_grad(set_to_none=True)
    print(f"SMOKE: gerçek {sum(p.numel() for p in model.parameters()):,} parametreli model; "
          f"shape/CE/weighted CE/backward/1 optimizer step OK; loss={loss.item():.4f}", flush=True)
    print(f"CPU tek microbatch smoke ölçümü: {elapsed:.2f}s; "
          f"{config.TRAIN_STEPS} optimizer step × {config.GRAD_ACCUM_STEPS} microbatch "
          f"kaba hesap={elapsed*config.TRAIN_STEPS*config.GRAD_ACCUM_STEPS/3600:.1f} saat "
          "(validation/checkpoint hariç; sürdürülebilir hız ölçülmedi)", flush=True)
    del logits, manual, plain, loss, before, optimizer
    gc.collect()

    probe = x[:1, :32]
    changed = probe.clone()
    changed[:, 16:] = (changed[:, 16:] + 37) % 256
    with torch.no_grad():
        original = model(probe)
        altered = model(changed)
        torch.testing.assert_close(original[:, :16], altered[:, :16], rtol=0, atol=1e-6)
        first, cache = model(probe[:, :16], use_cache=True)
        rest, cache = model(probe[:, 16:], past_key_values=cache, use_cache=True)
        torch.testing.assert_close(original, torch.cat((first, rest), dim=1), rtol=1e-4, atol=1e-5)
        _, one_cache = model(probe[:, :31], use_cache=True)
        last, one_cache = model(probe[:, 31:], past_key_values=one_cache, use_cache=True)
        torch.testing.assert_close(original[:, -1:], last, rtol=1e-4, atol=1e-5)
    print("SMOKE: gerçek 50M KV prefill/incremental logits eşitliği OK", flush=True)
    embedded = model.token_embedding(probe).detach().requires_grad_(True)
    hidden = embedded + model.position_embedding(torch.arange(probe.shape[1]))
    for block in model.blocks:
        hidden = block(hidden)
    model.lm_head(model.final_norm(hidden))[:, 15].sum().backward()
    assert embedded.grad[:, 16:].abs().max().item() == 0
    print("SMOKE: gerçek 50M modelde causal prefix ve geleceğe sıfır gradient OK", flush=True)
    del model, embedded, hidden, original, altered, first, rest, last, cache, one_cache
    gc.collect()

    tiny_config = dict(config.model_config(), CONTEXT_SIZE=32, EMBEDDING_DIM=32,
                       NUM_HEADS=4, NUM_LAYERS=2, FFN_DIM=128)
    tiny = KTYModel(tiny_config).eval()
    opt = torch.optim.AdamW(tiny.parameters(), lr=config.LEARNING_RATE)
    tx, ty, tw = make_batch(data, 32, 2, "cpu")
    sequence_loss(tiny(tx), ty, tw).backward()
    opt.step()
    buffer = io.BytesIO()
    torch.save(checkpoint_payload(tiny, opt, 1, data), buffer)
    buffer.seek(0)
    clone, restored = load_model_checkpoint(buffer, "cpu")
    clone.eval()
    assert clone.chat_format == config.CHAT_FORMAT
    assert clone.reasoning_tag == config.REASONING_TAG and clone.end_marker == config.END_MARKER
    clone_opt = torch.optim.AdamW(clone.parameters(), lr=config.LEARNING_RATE)
    clone_opt.load_state_dict(restored["optimizer_state_dict"])
    assert restored["data_rng_state"] == data.rng.getstate()
    assert restored["dataset_fingerprint"] == data.fingerprint
    for current, current_opt in ((tiny, opt), (clone, clone_opt)):
        current_opt.zero_grad(set_to_none=True)
        sequence_loss(current(tx), ty, tw).backward()
        current_opt.step()
    for a, b in zip(tiny.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    print("SMOKE: aynı model sınıfının küçük test konfigürasyonunda checkpoint save/load, "
          "optimizer+RNG resume eşitliği OK (RAM sınırı için)", flush=True)
    full = KTYModel(tiny_config)
    accumulated = KTYModel(tiny_config)
    accumulated.load_state_dict(full.state_dict())
    batch = 2 * config.GRAD_ACCUM_STEPS
    ax, ay, _ = make_batch(data, 32, batch, "cpu")
    sequence_loss(full(ax), ay).backward()
    for i in range(config.GRAD_ACCUM_STEPS):
        part = slice(2*i, 2*i+2)
        (sequence_loss(accumulated(ax[part]), ay[part]) / config.GRAD_ACCUM_STEPS).backward()
    for left, right in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
    assert 0 < learning_rate(1) <= learning_rate(config.WARMUP_STEPS)
    assert learning_rate(config.WARMUP_STEPS) == config.LEARNING_RATE
    assert abs(learning_rate(config.TRAIN_STEPS) - config.MIN_LEARNING_RATE) < 1e-12
    try:
        save_checkpoint(BASE_DIR / "forbidden-smoke.pt", tiny, opt, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Checkpoint dizin sınırı başarısız")
    print("SMOKE: gradient accumulation, LR warmup/cosine ve kayıt dizin sınırı OK", flush=True)
    tiny.eval()
    state = data.rng.getstate()
    first = evaluate(tiny, data)
    assert state == data.rng.getstate() and not tiny.training
    second = evaluate(tiny, data)
    assert first == second and all(math.isfinite(v) for v in first.values())
    print("SMOKE: sabit validation batch'leri, split izolasyonu ve RNG korunması OK", flush=True)
    for name in ("kty-1.0.pt", "kty-1.0-v2.pt", "kty-1.0-v3.pt"):
        path = BASE_DIR / "checkpoints" / name
        if path.exists():
            old_model, old_checkpoint = load_model_checkpoint(path)
            with torch.no_grad():
                result = old_model(probe[:, :min(16, old_model.context_size)])
            assert result.shape[-1] == 256 and torch.isfinite(result).all()
            try:
                save_checkpoint(path, tiny, opt, 1)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Checkpoint overwrite engeli başarısız")
            del old_model, old_checkpoint
            print(f"SMOKE: {name} strict load/forward/overwrite koruması OK", flush=True)
    from generate import generation_smoke_test
    generation_smoke_test()
    data.rng.setstate(saved_rng)
    print("SMOKE: TÜM TESTLER GEÇTİ; büyük model checkpointi oluşturulmadı.", flush=True)


def report(model, data, steps):
    params = sum(p.numel() for p in model.parameters())
    print("\nKTY-1.0 | sıfırdan byte decoder-only Transformer")
    print("Device: cpu; CPU threads:", torch.get_num_threads())
    print("Parametre:", params, "| Mimari:", model.config)
    print("Head boyutu 64; 10 blok ve 4× MLP, yalnız parametre şişirmek için seçilmedi.")
    print("Toplam chat/corpus byte:", data.numel)
    for name, stat in data.stats.items():
        print(f"  {name}: {stat['records']} kayıt/corpus satırı; train={stat['train']}, "
              f"validation={stat['validation']}, atlanan tam tekrar={stat['skipped_duplicates']}; "
              f"{stat['bytes']} byte; örnekleme ağırlığı={config.DATASET_WEIGHTS[name]:.0%}")
    print(f"Context={model.context_size}; microbatch={config.BATCH_SIZE}; "
          f"accumulation={config.GRAD_ACCUM_STEPS}; effective batch="
          f"{config.BATCH_SIZE*config.GRAD_ACCUM_STEPS}; steps={steps}")
    print(f"LR: warmup={config.WARMUP_STEPS}, max={config.LEARNING_RATE}, "
          f"min={config.MIN_LEARNING_RATE}; validation her {config.EVAL_INTERVAL} step")
    print("User tokenlarına ağırlık:", config.USER_LOSS_WEIGHT, "; cevap tokenlarına: 1.0")
    print("Yeni eğitim rastgele ağırlıklardan başlar; 2.6M v3 ağırlıkları 50M'e dönüştürülmez.")
    print("Reasoning: [Çözümü açıkla] girdisi -> kısa Çözüm / Cevap açıklaması; "
          "yetenek henüz eğitilip ölçülmedi.")
    print("Bir FP32 AdamW checkpointi yaklaşık", round(params*12/1024**2), "MiB.")
    checkpoint_count = (steps-1)//config.CHECKPOINT_INTERVAL + 1
    print(f"Ara + final checkpoint bütçesi yaklaşık {checkpoint_count*params*12/1024**3:.1f} GiB; "
          "eski checkpointler silinmez.", flush=True)


def evaluate_answers(path):
    from generate import generate_reply
    model, checkpoint = load_model_checkpoint(output_path(path), "cpu")
    if checkpoint.get("checkpoint_version", 0) >= 4:
        print("50M checkpoint cevap kontrolü")
    del checkpoint
    questions = [
        ("selam", None), ("merhaba", None), ("nasılsın", None),
        ("Sen kimsin?", "KTY-1.0"), ("Adın ne?", "KTY-1.0"),
        ("2+2", "4"), ("3+5", "8"), ("10-4", "6"), ("5×6", "30"), ("20÷4", "5"),
    ]
    passed = 0
    for question, expected in questions:
        answer = generate_reply(model, question, max_new_tokens=128, greedy=True)
        correct = expected is not None and (answer.strip().rstrip(".") == expected
                                           or expected == "KTY-1.0" and expected in answer)
        passed += int(correct)
        print(f"Soru: {question}\nKTY: {answer}\n"
              f"Kontrol: {'insan değerlendirmesi' if expected is None else 'OK' if correct else 'YANLIŞ'}",
              flush=True)
    print(f"Kimlik/kısa aritmetik exact kontrol: {passed}/7; sohbet ayrıca değerlendirilmeli.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Doğrulama ve smoke; eğitim başlatmaz")
    parser.add_argument("--resume", type=Path, help="Aynı 50M v4 mimarisinden devam et")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--evaluate-checkpoint", type=Path, help="Yalnız gerçek cevapları değerlendir")
    args = parser.parse_args()
    positive = ("CPU_THREADS", "BATCH_SIZE", "TRAIN_STEPS", "GRAD_ACCUM_STEPS",
                "EVAL_BATCHES", "EVAL_INTERVAL", "CHECKPOINT_INTERVAL")
    if any(getattr(config, k) <= 0 for k in positive):
        raise ValueError("Eğitim/batch/interval değerleri pozitif olmalı.")
    if not 0 < config.VALIDATION_FRACTION < 0.5 or not 0 < config.USER_LOSS_WEIGHT <= 1:
        raise ValueError("Validation fraction (0,0.5), user loss weight (0,1] olmalı.")
    if not 0 < config.MIN_LEARNING_RATE <= config.LEARNING_RATE or config.WARMUP_STEPS >= config.TRAIN_STEPS:
        raise ValueError("Learning rate sınırları/warmup/step hedefi geçersiz.")
    if any(v <= 0 for v in config.DATASET_WEIGHTS.values()):
        raise ValueError("Bütün veri kaynaklarının örnekleme ağırlığı pozitif olmalı.")
    torch.set_num_threads(config.CPU_THREADS)
    if args.evaluate_checkpoint:
        evaluate_answers(args.evaluate_checkpoint)
        return
    data, _ = load_training_data()
    smoke_test(data)
    torch.manual_seed(config.SEED)
    start_step = 0
    if args.resume:
        model, checkpoint = load_model_checkpoint(output_path(args.resume), "cpu")
        if (not isinstance(model, KTYModel) or model.config != config.model_config()
                or checkpoint.get("chat_format") != config.CHAT_FORMAT
                or checkpoint.get("end_marker") != config.END_MARKER
                or checkpoint.get("reasoning_tag") != config.REASONING_TAG):
            raise ValueError("Resume yalnız aynı 50M v4 mimarisiyle yapılabilir; eski model korunuyor.")
        if checkpoint.get("dataset_fingerprint") != data.fingerprint:
            raise ValueError("Resume verisi değişmiş; eski eğitim sözleşmesiyle eşleşmiyor.")
        for key in ("BATCH_SIZE", "GRAD_ACCUM_STEPS", "USER_LOSS_WEIGHT", "DATASET_WEIGHTS",
                    "VALIDATION_FRACTION", "SEED", "WEIGHT_DECAY"):
            if checkpoint["training_config"][key] != getattr(config, key):
                raise ValueError(f"Resume training config uyuşmazlığı: {key}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                      weight_decay=config.WEIGHT_DECAY)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        data.rng.setstate(checkpoint["data_rng_state"])
        del checkpoint
    else:
        model = KTYModel().cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                      weight_decay=config.WEIGHT_DECAY)
    report(model, data, config.TRAIN_STEPS)
    if args.check:
        print("EĞİTİM BAŞLATILMADI. Başlat: python train.py", flush=True)
        return
    output = output_path(args.checkpoint)
    if output.exists():
        raise FileExistsError(f"Checkpoint korunuyor: {output}. --checkpoint ile yeni ad seç.")
    if start_step >= config.TRAIN_STEPS:
        raise ValueError("Resume step >= TRAIN_STEPS; config.py içindeki hedefi artır.")
    for step in range(start_step + 1, config.TRAIN_STEPS):
        if step % config.CHECKPOINT_INTERVAL == 0:
            intermediate = output.with_name(f"{output.stem}-step{step:06d}.pt")
            if intermediate.exists():
                raise FileExistsError(f"Ara checkpoint korunuyor: {intermediate}; yeni çıkış adı seç.")
    model.train()
    start = time.perf_counter()
    metrics = {}
    window_loss = 0.0
    window_count = 0
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        if not stop_requested:
            print("\nKesme isteği alındı; mevcut optimizer adımı tamamlanıp kayıt alınacak.",
                  flush=True)
        stop_requested = True

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    print("Eğitim başlıyor; hazır/pretrained ağırlık kullanılmıyor.", flush=True)
    try:
        for step in range(start_step + 1, config.TRAIN_STEPS + 1):
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(config.GRAD_ACCUM_STEPS):
                x, y, weights = make_batch(data, model.context_size, config.BATCH_SIZE, "cpu")
                loss = sequence_loss(model(x), y, weights)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Step {step}: sonlu olmayan loss.")
                (loss / config.GRAD_ACCUM_STEPS).backward()
                step_loss += loss.detach().item() / config.GRAD_ACCUM_STEPS
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            if not torch.isfinite(norm):
                raise RuntimeError(f"Step {step}: sonlu olmayan gradient.")
            optimizer.step()
            if stop_requested:
                path = output.with_name(f"{output.stem}-interrupted-step{step:06d}.pt")
                save_checkpoint(path, model, optimizer, step, data, metrics)
                print("Eğitim kesildi; resume checkpoint:", path, flush=True)
                return
            window_loss += step_loss
            window_count += 1
            if step == start_step + 1 or step % 50 == 0:
                elapsed = time.perf_counter() - start
                speed = (step - start_step) / elapsed
                eta = (config.TRAIN_STEPS-step)/max(speed, 1e-9)/3600
                print(f"step={step}/{config.TRAIN_STEPS} train_loss={window_loss/window_count:.4f} "
                      f"lr={lr:.7f} süre={elapsed/60:.1f}dk step/s={speed:.3f} ETA={eta:.1f}saat",
                      flush=True)
                window_loss = 0.0
                window_count = 0
            if step % config.EVAL_INTERVAL == 0 or step == config.TRAIN_STEPS:
                metrics = evaluate(model, data)
                metrics["validation_step"] = step
                print(f"VALIDATION step={step} CE={metrics['validation_loss']:.4f} "
                      f"weighted_CE={metrics['validation_weighted_loss']:.4f}", flush=True)
            if step % config.CHECKPOINT_INTERVAL == 0 and step != config.TRAIN_STEPS:
                path = output.with_name(f"{output.stem}-step{step:06d}.pt")
                save_checkpoint(path, model, optimizer, step, data, metrics)
                print("Ara checkpoint:", path.name, flush=True)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    save_checkpoint(output, model, optimizer, config.TRAIN_STEPS, data, metrics)
    print("KTY-1.0 eğitimi tamamlandı. Yeni checkpoint:", output, flush=True)
    print(f'Cevap testi: python train.py --evaluate-checkpoint "{output}"', flush=True)


if __name__ == "__main__":
    main()
