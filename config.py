VOCAB_SIZE = 256
CONTEXT_SIZE = 768
EMBEDDING_DIM = 640
NUM_HEADS = 10
NUM_LAYERS = 10
FFN_DIM = 2560
LEARNING_RATE = 0.0002
MIN_LEARNING_RATE = 0.00002
TRAIN_STEPS = 20000
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EVAL_INTERVAL = 500
EVAL_BATCHES = 4
CHECKPOINT_INTERVAL = 2000
VALIDATION_FRACTION = 0.05
USER_LOSS_WEIGHT = 0.35
CPU_THREADS = 4
MAX_NEW_TOKENS = 512
REASONING_MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6
TOP_K = 10
SEED = 42
CHECKPOINT_NAME = "kty-1.0-v4-50m.pt"
CHAT_FORMAT = "kty-chat-v4"
END_MARKER = "<|son|>"
REASONING_TAG = "[Çözümü açıkla]"
DATASET_WEIGHTS = {
    "train.txt": 0.01,
    "kty_chat.jsonl": 0.02,
    "sertlik.jsonl": 0.04,
    "sohbet.jsonl": 0.35,
    "matematik.jsonl": 0.20,
    "python.jsonl": 0.18,
    "reasoning.jsonl": 0.20,
}


def model_config():
    return {name: globals()[name] for name in (
        "VOCAB_SIZE", "CONTEXT_SIZE", "EMBEDDING_DIM", "NUM_HEADS",
        "NUM_LAYERS", "FFN_DIM",
    )}
    