#!/usr/bin/env python3
# Refactored PPO-KL training script using ChemQ3MTP module

import os
import torch
from tqdm import tqdm
from FastChemTokenizer import FastChemTokenizerSelfies
from ChemQ3MTP import ChemQ3MTP, CurriculumManager

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- Load tokenizer ---
    tokenizer = FastChemTokenizerSelfies.from_pretrained("../selftok_core")

    # --- Load model ---
    model = ChemQ3MTP.from_pretrained("./pretrained/sample-e1-mtp")
    model.tokenizer = tokenizer
    model.to(device)

    # --- RL fine-tuning setup ---
    print("\n🎯 Phase 2: RL Fine-tuning with PPO + Curriculum Learning")
    model.set_mtp_training(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    curriculum = CurriculumManager(start_len=10, max_len=80, step_increase=5, steps_per_level=100)
    baseline = None
    gamma = 0.95

    # Dummy input (BOS-only batch)
    batch_size = 4
    dummy_input = tokenizer([tokenizer.bos_token] * batch_size, return_tensors="pt", padding=True)
    input_ids = dummy_input.input_ids.to(device)

    # Training config
    total_steps = 1000
    checkpoint_steps = {total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps}
    checkpoint_dir = "./ppo_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- RL Training Loop with tqdm ---
    for step in tqdm(range(total_steps), desc="RL Training"):
        max_new_tokens = curriculum.get_max_new_tokens()

        # === PPO Rollout ===
        with torch.no_grad():
            selfies_list, old_log_probs, _, old_action_probs = model.generate_with_logprobs(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                return_probs=True
            )
            old_log_probs = old_log_probs.detach()
            old_action_probs = old_action_probs.detach()

        # === PPO Update ===
        ppo_result = model.ppo_step(
            input_ids=input_ids,
            old_log_probs=old_log_probs,
            old_action_probs=old_action_probs,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            validity_weight=1.0,
            lipinski_weight=1.0,
            entropy_weight=0.01,
            clip_epsilon=0.2,
            baseline=baseline,
        )

        loss = ppo_result['loss']
        optimizer.zero_grad(set_to_none=True)  # slightly more efficient than zeroing
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # === Update baseline ===
        reward_tensor = torch.tensor(ppo_result['avg_reward'], device=device)
        baseline = reward_tensor if baseline is None else gamma * baseline + (1 - gamma) * reward_tensor

        # Curriculum update
        curriculum.step()

        # Checkpointing
        if (step + 1) in checkpoint_steps:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step+1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            torch.save({
                'step': step + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'baseline': baseline.item(),
                'curriculum_state': {
                    'current_max_len': curriculum.current_max_len,
                    'step_counter': curriculum.step_counter
                }
            }, os.path.join(checkpoint_path, 'training_state.pt'))
            print(f"\n💾 Checkpoint saved at step {step+1} -> {checkpoint_path}")

         # Logging every 50 steps
        if step % 50 == 0:
            print(f"\n[RL Step {step}] "
                  f"Loss={loss.item():.4f} | "
                  f"Valid={ppo_result['validity_rate']:.3f} | "
                  f"Lipinski={ppo_result['lipinski_score']:.3f} | "
                  f"Bio={ppo_result['biological_score']:.3f} | "
                  f"Neutral={ppo_result['neutrality_rate']:.3f} | "
                  f"Complex={ppo_result['complexity_score']:.3f} | "
                  f"Entropy={ppo_result['entropy']:.3f} | "
                  f"EntropyW={ppo_result['entropy_weight']:.4f}")

            sample_selfies = ppo_result['generated_selfies'][0][:100]
            sample_smiles = ppo_result['generated_smiles'][0] or "Invalid"
            print(f"  Sample SELFIES: {sample_selfies}")
            print(f"  Sample SMILES: {sample_smiles}")


    print("🎉 Training complete!")

if __name__ == "__main__":
    main()
