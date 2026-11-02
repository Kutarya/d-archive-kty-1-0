import argparse
import codecs
import math
from pathlib import Path

import torch

import config
from model import KTYModel, load_model_checkpoint
from tokenizer import KTYTokenizer

BASE_DIR = Path(__file__).resolve().parent
STOP_MARKERS = ("Kullanıcı:".encode("utf-8"), b"KTY:")
DEFAULT_TEMPERATURE = min(config.TEMPERATURE, 0.6)
DEFAULT_TOP_K = min(config.TOP_K, 10) if config.TOP_K else 0


class UTF8Constraint:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")

    def allowed(self, remaining):
        state = self.decoder.getstate()
        pending = state[0]
        allowed = []
        for value in range(256):
            probe = codecs.getincrementaldecoder("utf-8")("strict")
            probe.setstate(state)
            try:
                probe.decode(bytes([value]), final=False)
            except UnicodeDecodeError:
                continue
            after = probe.getstate()[0]
            if after:
                needed = (2 if after[0] < 0xE0 else 3 if after[0] < 0xF0 else 4) - len(after)
                if needed > remaining - 1:
                    continue
            if not pending and (value < 32 and value not in (9, 10) or value == 127):
                continue
            allowed.append(value)
        return allowed

    def advance(self, value):
        self.decoder.decode(bytes([value]), final=False)


def turn_boundary(generated):
    cuts = []
    for marker in STOP_MARKERS:
        start = 0
        while True:
            cut = generated.find(marker, start)
            if cut < 0:
                break
            prefix = generated[:cut]
            line_start = prefix.rsplit(b"\n", 1)[-1]
            if not line_start.strip() and prefix.count(b"```") % 2 == 0:
                cuts.append(cut)
            start = cut + len(marker)
    return min(cuts) if cuts else None


def generate_reply(model, message, max_new_tokens=config.MAX_NEW_TOKENS,
                   temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K, greedy=False,
                   utf8_constraint=True, reasoning=False, use_cache=True):
    if max_new_tokens <= 0 or not math.isfinite(temperature) or temperature <= 0 or not 0 <= top_k <= 256:
        raise ValueError("max_new_tokens/temperature pozitif, top_k 0..256 olmalı.")
    if not message.strip():
        raise ValueError("Kullanıcı mesajı boş olamaz.")
    tokenizer = KTYTokenizer()
    v4 = getattr(model, "chat_format", "legacy") == config.CHAT_FORMAT
    if reasoning and not v4:
        raise ValueError("Reasoning biçimi yalnız yeni v4 checkpointinde desteklenir.")
    user = message.strip()
    if reasoning and not user.startswith(model.reasoning_tag):
        user = f"{model.reasoning_tag} {user}"
    prompt = f"Kullanıcı: {user}\nKTY:" + (" " if v4 else "")
    end_marker = ("\n" + model.end_marker).encode("utf-8") if v4 else None
    tokens = tokenizer.encode(prompt)
    generated = bytearray()
    constraint = UTF8Constraint()
    device = next(model.parameters()).device
    model.eval()
    cached = None
    cache_enabled = use_cache and getattr(model, "supports_kv_cache", False)
    with torch.inference_mode():
        for index in range(max_new_tokens):
            if cache_enabled and cached is not None and cached[0][0].shape[2] < model.context_size:
                x = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                output, cached = model(x, past_key_values=cached, use_cache=True)
            else:
                x = torch.tensor([tokens[-model.context_size:]], dtype=torch.long, device=device)
                if cache_enabled:
                    output, cached = model(x, use_cache=True)
                else:
                    output = model(x)
            logits = output[0, -1].clone()
            if utf8_constraint:
                valid = constraint.allowed(max_new_tokens - index)
                mask = torch.ones(256, dtype=torch.bool, device=device)
                mask[valid] = False
                logits.masked_fill_(mask, -torch.inf)
            if greedy:
                token = logits.argmax().item()
            else:
                logits /= temperature
                if top_k:
                    keep = torch.topk(logits, top_k).indices
                    selected = torch.full_like(logits, -torch.inf)
                    selected[keep] = logits[keep]
                    logits = selected
                token = torch.multinomial(logits.softmax(-1), 1).item()
            tokens.append(token)
            generated.append(token)
            if utf8_constraint:
                constraint.advance(token)
            if end_marker and generated.endswith(end_marker):
                generated = generated[:-len(end_marker)]
                break
            cut = turn_boundary(generated)
            if cut is not None:
                generated = generated[:cut]
                break
            inside_code = generated.count(b"```") % 2 == 1
            if not v4 and generated.endswith(b"\n\n") and not inside_code:
                generated = generated[:-2]
                break
    if utf8_constraint:
        return bytes(generated).decode("utf-8", errors="strict").strip()
    return tokenizer.decode(generated).strip()


def generation_smoke_test():
    class ScriptedModel(torch.nn.Module):
        def __init__(self, continuation, v4=False):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.context_size = config.CONTEXT_SIZE
            self.continuation = list(continuation)
            self.index = 0
            self.chat_format = config.CHAT_FORMAT if v4 else "legacy"
            self.reasoning_tag = config.REASONING_TAG if v4 else ""
            self.end_marker = config.END_MARKER if v4 else ""
            self.first_input = None

        def forward(self, tokens):
            if self.first_input is None:
                self.first_input = bytes(tokens[0].tolist()).decode("utf-8")
            value = self.continuation[min(self.index, len(self.continuation) - 1)]
            self.index += 1
            logits = torch.full((*tokens.shape, 256), -100.0)
            logits[:, -1, value] = 100.0
            return logits

    for boundary in ("\nKullanıcı: başka", "\nKTY: başka", "\n\n"):
        reply = generate_reply(ScriptedModel((" Selam, nasılsın?" + boundary).encode()),
                               "selam", max_new_tokens=100, greedy=True)
        assert reply == "Selam, nasılsın?"
    for text in ("çğıöşü İĞÜ", "🙂", "kısa"):
        reply = generate_reply(ScriptedModel(text.encode()), "selam",
                               max_new_tokens=len(text.encode()), greedy=True)
        assert reply == text
    code = "```python\nimport math\n\nprint(math.sqrt(9))\n```"
    reply = generate_reply(ScriptedModel((code + "\n\n").encode()), "Python kodu yaz",
                           max_new_tokens=100, greedy=True)
    assert reply == code
    text = 'İsim etiketi KTY: olarak yazılır.\n```python\nprint("KTY:")\n```'
    reply = generate_reply(ScriptedModel((text + "\nKullanıcı: devam").encode()), "Etiketi göster",
                           max_new_tokens=180, greedy=True)
    assert reply == text
    text = "Kısa açıklama.\n\n```python\nprint('Selam')\n```"
    scripted = ScriptedModel((text + "\n" + config.END_MARKER).encode(), v4=True)
    assert generate_reply(scripted, "Python örneği", max_new_tokens=180, greedy=True) == text
    assert scripted.first_input == "Kullanıcı: Python örneği\nKTY: "
    text = "Çözüm: 2 + 2 = 4.\nCevap: 4"
    scripted = ScriptedModel((text + "\n" + config.END_MARKER).encode(), v4=True)
    assert generate_reply(scripted, "2+2", max_new_tokens=180, greedy=True,
                          reasoning=True) == text
    assert scripted.first_input == f"Kullanıcı: {config.REASONING_TAG} 2+2\nKTY: "
    model = ScriptedModel(bytes([0xFF, 0x80, 0xF4, 0xBF, 0x0A]))
    generate_reply(model, "selam", max_new_tokens=12, greedy=True).encode("utf-8")
    tiny = KTYModel(dict(config.model_config(), CONTEXT_SIZE=32, EMBEDDING_DIM=32,
                         NUM_HEADS=4, NUM_LAYERS=2, FFN_DIM=128))
    tiny.chat_format, tiny.reasoning_tag, tiny.end_marker = (config.CHAT_FORMAT,
                                                           config.REASONING_TAG, config.END_MARKER)
    cached_reply = generate_reply(tiny, "selam", max_new_tokens=48, greedy=True)
    plain_reply = generate_reply(tiny, "selam", max_new_tokens=48, greedy=True, use_cache=False)
    assert cached_reply == plain_reply
    print("SMOKE: v3/v4 generation, EOS, reasoning prompt, kod boş satırları ve UTF-8 OK",
          flush=True)
    print("SMOKE: KV cache ve context dolunca yeniden hesaplama, aynı greedy cevap OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description="KTY-1.0 terminal sohbeti")
    parser.add_argument("--checkpoint", type=Path, help="Varsayılan: v4-50m; yoksa mevcut v3")
    parser.add_argument("--greedy", action="store_true", help="Sampling yerine en olası byte'ı seç")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--reasoning", action="store_true", help="v4 kısa çözüm açıklaması biçimini iste")
    parser.add_argument("--no-cache", action="store_true", help="KV cache olmadan karşılaştırma yap")
    parser.add_argument("--prompt", help="Tek soruya cevap verip çık")
    parser.add_argument("--seed", type=int, help="Sampling denemesini tekrarlanabilir yap")
    parser.add_argument("--raw-bytes", action="store_true",
                        help="UTF-8 kısıtını kaldırır; bozuk karakterler görünür olabilir")
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = (config.REASONING_MAX_NEW_TOKENS if args.reasoning
                               else config.MAX_NEW_TOKENS)
    if args.checkpoint is None:
        v4_path = BASE_DIR / "checkpoints" / config.CHECKPOINT_NAME
        args.checkpoint = (v4_path if v4_path.is_file() else
                           BASE_DIR / "checkpoints" / "kty-1.0-v3.pt")
    if (args.max_new_tokens <= 0 or not math.isfinite(args.temperature)
            or args.temperature <= 0 or not 0 <= args.top_k <= 256):
        parser.error("max-new-tokens/temperature pozitif, top-k 0..256 olmalı.")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(config.CPU_THREADS)
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint bulunamadı: {args.checkpoint}. Önce python train.py çalıştır.")
    model, checkpoint = load_model_checkpoint(args.checkpoint, "cpu")
    checkpoint.pop("model_state_dict", None)
    checkpoint.pop("optimizer_state_dict", None)
    if args.reasoning and getattr(model, "chat_format", "legacy") != config.CHAT_FORMAT:
        parser.error("Reasoning için önce v4-50m checkpointini eğitmelisin; mevcut v3 korunuyor.")
    if not isinstance(model, KTYModel):
        print("Eski tek-head checkpoint kullanılıyor; byte tokenizer korunuyor.")
    print(f"KTY-1.0 yüklendi | {args.checkpoint.name} | CPU", flush=True)
    if isinstance(model, KTYModel):
        print(f"Eğitim adımı: {checkpoint.get('step', 'bilinmiyor')} | "
              f"Parametre: {sum(p.numel() for p in model.parameters()):,} | "
              f"Context: {model.context_size} byte", flush=True)
    if args.checkpoint.name == "kty-1.0-v3.pt":
        print("Mevcut 2.6M v3 kullanılıyor; yeni 50M v4 henüz yüklenmedi.")
    settings = ("greedy" if args.greedy else
                f"sampling, temperature={args.temperature}, top_k={args.top_k}")
    settings += f", max_new_tokens={args.max_new_tokens}"
    if args.reasoning:
        settings += ", kısa çözüm açıklaması"

    def answer(message):
        reply = generate_reply(model, message, args.max_new_tokens, args.temperature,
                               args.top_k, args.greedy, not args.raw_bytes, args.reasoning,
                               not args.no_cache)
        print("KTY: " + (reply if reply else "[Model boş cevap üretti.]"), flush=True)

    if args.prompt is not None:
        if not args.prompt.strip():
            parser.error("--prompt boş olamaz.")
        answer(args.prompt)
        return
    print(f"Mod: {settings}")
    print("Mesajını yaz. /cikis: çıkış | /ayarlar: ayarlar | /yardim: yardım")
    try:
        while True:
            message = input("\nSen: ")
            if not message.strip():
                continue
            command = message.strip().casefold()
            if command in ("/cikis", "/çıkış", "/exit"):
                break
            if command == "/ayarlar":
                print(settings)
                continue
            if command == "/yardim":
                print("Her mesaj bağımsız yanıtlanır; önceki turlar modele gönderilmez.\n"
                      "Farklı modla başlat: python generate.py --greedy\n"
                      "v4 çözüm açıklaması: python generate.py --reasoning\n"
                      "Tek soru: python generate.py --prompt selam --seed 42")
                continue
            answer(message)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
