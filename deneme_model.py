import torch
import torch.nn as nn
import torch.nn.functional as F
import math

prompt = input("Please enter your prompt: ")

context_size = 256
embedding_dim = 64


class KTYMiniModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            256,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            context_size,
            embedding_dim
        )

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.KTYcevap = nn.Linear(
            embedding_dim,
            256
        )

    def forward(self, x):
        B, T = x.shape

        token_embedding = self.embedding(x)

        pozisyonlar = torch.arange(
            T,
            device=x.device
        )

        position_embedding = self.position_embedding(
            pozisyonlar
        )

        x = token_embedding + position_embedding

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(embedding_dim)

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        x = attention_weights @ V

        son_context = x[:, -1, :]

        logits = self.KTYcevap(
            son_context
        )

        return logits, attention_weights


model = KTYMiniModel()


tokenlar = list(
    prompt.encode("utf-8")
)

if len(tokenlar) <= context_size:
    print(
        "HATA: Yazdığın metin context_size'dan uzun olmalı."
    )
    exit()


X = []
Y = []

for i in range(
    len(tokenlar) - context_size
):
    context = tokenlar[
        i:i + context_size
    ]

    hedef = tokenlar[
        i + context_size
    ]

    X.append(context)
    Y.append(hedef)


x = torch.tensor(
    X,
    dtype=torch.long
)

y = torch.tensor(
    Y,
    dtype=torch.long
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01
)


for adim in range(500):
    KTYcevap, attention = model(x)

    loss = F.cross_entropy(
        KTYcevap,
        y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if adim % 100 == 0:
        print(
            "Adım:",
            adim,
            "Loss:",
            loss.item()
        )


KTYcevap, attention = model(x)

tahminler = torch.argmax(
    KTYcevap,
    dim=1
)


print("\nX:")
print(x)

print("\nY:")
print(y)

print("\nModelin tahminleri:")
print(tahminler)

print("\nDoğru cevaplar:")
print(y)


print(
    "\nToplam parametre:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


def metin_uret(model, baslangic, uzunluk=50):
    model.eval()

    uretilen = list(
        baslangic.encode("utf-8")
    )

    if len(uretilen) < context_size:
        print(
            "Başlangıç metni en az",
            context_size,
            "byte olmalı."
        )
        return ""

    with torch.no_grad():
        for _ in range(uzunluk):
            context = uretilen[
                -context_size:
            ]

            x_uret = torch.tensor(
                [context],
                dtype=torch.long
            )

            logits, attention = model(
                x_uret
            )

            sonraki_byte = torch.argmax(
                logits[0]
            ).item()

            uretilen.append(
                sonraki_byte
            )

    return bytes(
        uretilen
    ).decode(
        "utf-8",
        errors="replace"
    )


baslangic = input(
    "\nKTY için başlangıç metni: "
)

sonuc = metin_uret(
    model,
    baslangic,
    uzunluk=60
)

print("\nKTY-1.0:")
print(sonuc)
