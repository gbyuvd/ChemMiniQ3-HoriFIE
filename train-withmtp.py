# ========================
#  Train with NTP + MTP
#  by gbyuvd
# ========================

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import math
from typing import List, Union, Optional, Tuple, Dict, Any
from transformers.tokenization_utils_base import BatchEncoding
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel
from datasets import load_dataset, DatasetDict
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from ranger21 import Ranger21
from tqdm.notebook import tqdm
from FastChemTokenizer import FastChemTokenizerSelfies
from ChemQ3MTP import ChemQ3MTP
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import TrainerCallback
import datetime

# ==============================
# Load external configuration
# ==============================
with open("config.json", "r") as f:
    CONFIG = json.load(f)

TRAINING_CFG = CONFIG["training"]
MODEL_CFG = CONFIG["model"]
GENERATION_CFG = CONFIG.get("generation", {})

# Training params
BATCH_SIZE = TRAINING_CFG["batch_size"]
NUM_EPOCHS = TRAINING_CFG["num_epochs"]
LEARNING_RATE = TRAINING_CFG["learning_rate"]
WEIGHT_DECAY = TRAINING_CFG["weight_decay"]
GRAD_ACCUM_STEPS = TRAINING_CFG["gradient_accumulation_steps"]
TOKENIZE_BATCH_SIZE = TRAINING_CFG["tokenize_batch_size"]
TRAIN_SPLIT_RATIO = TRAINING_CFG["train_split_ratio"]
VAL_SPLIT_RATIO = TRAINING_CFG["val_split_ratio"]
TEST_SPLIT_RATIO = TRAINING_CFG["test_split_ratio"]
INCLUDE_FOR_METRICS = TRAINING_CFG.get("include_for_metrics", ["input_ids", "attention_mask", "labels"])
# ==============================

class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_losses.txt", with_timestamp=False):
        self.log_file = log_file
        self.with_timestamp = with_timestamp
        with open(self.log_file, "w") as f:
            if self.with_timestamp:
                f.write("time\tstep\tloss\teval_loss\n")
            else:
                f.write("step\tloss\teval_loss\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")

        with open(self.log_file, "a") as f:
            if self.with_timestamp:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts}\t{step}\t{loss if loss is not None else ''}\t{eval_loss if eval_loss is not None else ''}\n")
            else:
                f.write(f"{step}\t{loss if loss is not None else ''}\t{eval_loss if eval_loss is not None else ''}\n")


def main():
    # --- Load the tokenizer ---
    tokenizer = FastChemTokenizerSelfies.from_pretrained("./selftok_core")

    out = tokenizer("[C] [=C] [Branch1]", return_tensors="pt")
    print(out.input_ids)
    print(out.attention_mask)
    out = out.to("cuda" if torch.cuda.is_available() else "cpu")
    print(out.input_ids.device)

    # --- Define config ---
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
        use_cache=False,
        **MODEL_CFG
    )

    model = ChemQ3MTP(config, num_future_tokens=3)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Enhanced model has {count_parameters(model):,} trainable parameters.")

    batch_size, seq_len = 2, 32
    dummy_input = torch.randint(
        low=0,
        high=len(tokenizer),
        size=(batch_size, seq_len),
        dtype=torch.long,
    )
    with torch.no_grad():
        outputs = model(dummy_input)
        logits = outputs.logits
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")

    print("Loading dataset...")
    dataset = load_dataset(
        'csv',
        data_files='./data/sample_all_14k.csv',
        split='train',
        streaming=True
    )

    print("Shuffling and splitting dataset...")
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10000)

    total_lines = 14000
    test_size = int(TEST_SPLIT_RATIO * total_lines)
    val_size = int(VAL_SPLIT_RATIO * total_lines)
    train_size = total_lines - test_size - val_size

    test_dataset = shuffled_dataset.take(test_size)
    remaining = shuffled_dataset.skip(test_size)
    val_dataset = remaining.take(val_size)
    train_dataset = remaining.skip(val_size)

    print(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")

    def tokenize_function(examples):
        batch_results = {"input_ids": [], "attention_mask": [], "labels": []}
        smiles_list = examples['SELFIES'] if isinstance(examples['SELFIES'], list) else [examples['SELFIES']]
        for smiles in smiles_list:
            tokenized = tokenizer(
                smiles,
                truncation=True,
                padding=False,
                max_length=MODEL_CFG["max_position_embeddings"],
                return_tensors=None,
                add_special_tokens=True
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = input_ids.copy()
            batch_results["input_ids"].append(input_ids)
            batch_results["attention_mask"].append(attention_mask)
            batch_results["labels"].append(labels)
        return batch_results

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=TOKENIZE_BATCH_SIZE, remove_columns=["SELFIES"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=TOKENIZE_BATCH_SIZE, remove_columns=["SELFIES"])

    class EnhancedDataCollator:
        def __init__(self, tokenizer, pad_to_multiple_of=8):
            self.tokenizer = tokenizer
            self.pad_to_multiple_of = pad_to_multiple_of
        def __call__(self, features):
            max_length = max(len(f["input_ids"]) for f in features)
            if self.pad_to_multiple_of:
                max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for feature in features:
                input_ids = feature["input_ids"]
                attention_mask = feature["attention_mask"]
                labels = feature["labels"]
                padding_length = max_length - len(input_ids)
                padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                padded_attention_mask = attention_mask + [0] * padding_length
                padded_labels = labels + [-100] * padding_length
                batch["input_ids"].append(padded_input_ids)
                batch["attention_mask"].append(padded_attention_mask)
                batch["labels"].append(padded_labels)
            batch = {key: torch.tensor(values, dtype=torch.long) for key, values in batch.items()}
            return batch

    data_collator = EnhancedDataCollator(tokenizer, pad_to_multiple_of=8)

    def create_enhanced_optimizer(model_params):
        num_batches_per_epoch = train_size // BATCH_SIZE
        optimizer_params = {
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'use_adabelief': True,
            'use_cheb': False,
            'use_warmup': True,
            'use_madgrad': True,
            'num_epochs': NUM_EPOCHS,
            'using_gc': True,
            'warmdown_active': True,
            'num_batches_per_epoch': num_batches_per_epoch
        }
        return Ranger21(model_params, **optimizer_params)

    from torch.optim.lr_scheduler import LambdaLR
    class EnhancedCustomTrainer(Trainer):
        def create_optimizer(self):
            self.optimizer = create_enhanced_optimizer(self.model.parameters())
            return self.optimizer
        def create_scheduler(self, num_training_steps, optimizer=None):
            if optimizer is None:
                optimizer = self.optimizer
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
            return self.lr_scheduler
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    steps_per_epoch = train_size // BATCH_SIZE
    total_steps = steps_per_epoch * NUM_EPOCHS

    training_args = TrainingArguments(
        output_dir='./chemq3minipret',
        max_steps=total_steps,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        logging_dir='./gptlo-1',
        logging_strategy="steps",
        logging_steps=max(1, steps_per_epoch // 4),
        eval_strategy="steps",
        eval_steps=max(1, steps_per_epoch // 4),
        save_strategy="steps",
        save_steps=steps_per_epoch,
        save_total_limit=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        prediction_loss_only=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        report_to=None,
        include_for_metrics=INCLUDE_FOR_METRICS,
    )

    print("Initializing enhanced trainer with MTP capabilities...")
    trainer = EnhancedCustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[LossLoggerCallback("training_losses.txt", with_timestamp=True)]
    )

    model.set_mtp_training(True)
    print(" MTP training mode enabled")

    print("Starting enhanced training with MTP and Horizon Loss...")
    try:
        print("\n Phase 1: Warmup with standard Causal LM...")
        model.set_mtp_training(False)
        warmup_steps = max(1, total_steps // 5)
        trainer.args.max_steps = warmup_steps
        trainer.train()
        print("\n Phase 2: Full MTP + Horizon Loss training...")
        model.set_mtp_training(True)
        trainer.args.max_steps = total_steps
        trainer.train(resume_from_checkpoint=True)
        print("Enhanced training completed successfully!")
        trainer.save_model("./enhanced-qwen3-final")
        tokenizer.save_pretrained("./enhanced-qwen3-final")
        training_config = {
            "model_type": "EnhancedQwen3ForCausalLM",
            "num_future_tokens": 3,
            "horizon_loss_enabled": True,
            "mtp_head_enabled": True,
            "training_phases": ["causal_lm_warmup", "mtp_horizon_training"],
            "total_parameters": count_parameters(model),
        }
        config_path = "./enhanced-qwen3-final/training_config.json"
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2)
        print(f" Enhanced model, tokenizer, and config saved!")
    except Exception as e:
        print(f"Enhanced training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nmTesting enhanced generation capabilities...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    try:
        print("\n--- Standard Generation Test ---")
        input_ids = tokenizer("<s> [C]", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.set_mtp_training(False)
            gen = model.generate(
                input_ids,
                max_length=GENERATION_CFG.get("max_length", 64),
                top_k=GENERATION_CFG.get("top_k", 50),
                top_p=GENERATION_CFG.get("top_p", 0.9),
                temperature=GENERATION_CFG.get("temperature", 0.8),
                do_sample=GENERATION_CFG.get("do_sample", True),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=GENERATION_CFG.get("num_return_sequences", 3),
            )
            for i, sequence in enumerate(gen):
                result = tokenizer.decode(sequence, skip_special_tokens=True)
                print(f"Generated SELFIES {i+1}: {result}")
        print("\n--- MTP Analysis Test ---")
        model.set_mtp_training(True)
        test_smiles = "[C]"
        test_input = tokenizer(test_smiles, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            outputs = model(**test_input)
            if hasattr(model.mtp_head, 'prediction_heads'):
                hidden_states = model.model(test_input['input_ids']).last_hidden_state
                mtp_outputs = model.mtp_head(hidden_states)
                print(f"Input SELFIES: {test_smiles}")
                print(f"Tokenized: {tokenizer.convert_ids_to_tokens(test_input['input_ids'][0].tolist())}")
                for i, (key, logits) in enumerate(mtp_outputs.items()):
                    top_tokens = torch.topk(logits[0], k=3, dim=-1)
                    print(f"\n{key} predictions:")
                    for pos in range(min(5, logits.size(1))):
                        pos_preds = []
                        for j in range(3):
                            token_id = top_tokens.indices[pos, j].item()
                            prob = torch.softmax(logits[0, pos], dim=-1)[token_id].item()
                            token = tokenizer.id_to_token.get(token_id, '<UNK>')
                            pos_preds.append(f"{token}({prob:.3f})")
                        print(f"  Position {pos}: {', '.join(pos_preds)}")
        print("\nEnhanced generation tests completed!")
    except Exception as e:
        print(f"Enhanced generation test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nEnhanced Model Analysis:")
    print(f"Total parameters: {count_parameters(model):,}")
    mtp_params = sum(p.numel() for p in model.mtp_head.parameters() if p.requires_grad)
    horizon_params = sum(p.numel() for p in model.horizon_loss.parameters() if p.requires_grad)
    base_params = count_parameters(model) - mtp_params - horizon_params
    print(f"Base model parameters: {base_params:,}")
    print(f"MTP head parameters: {mtp_params:,}")
    print(f"Horizon loss parameters: {horizon_params:,}")
    print(f"Enhancement overhead: {((mtp_params + horizon_params) / base_params * 100):.2f}%")
    print(f"\n Enhanced Model Architecture:")
    print(f"- Base Model: Qwen3 with {config.num_hidden_layers} layers")
    print(f"- Hidden Size: {config.hidden_size}")
    print(f"- Attention Heads: {config.num_attention_heads}")
    print(f"- Vocab Size: {config.vocab_size}")
    print(f"- MTP Future Tokens: {model.mtp_head.num_future_tokens}")
    print(f"- Horizon Loss Weights: Learnable")
    print(f"- Training Mode: {'MTP + Horizon Loss' if model.use_mtp_training else 'Standard Causal LM'}")
    print("\n Enhanced training pipeline completed successfully!")

if __name__ == "__main__":
    main()