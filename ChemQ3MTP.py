# ========================
#  ChemQ3-MTP
#  MODEL COMPONENTS 
#  by gbyuvd
# ========================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Union, Optional, Tuple, Dict, Any
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import selfies as sf
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')      # suppress all SMILES parse messages
import json
from typing import List, Union, Optional, Tuple
from transformers.tokenization_utils_base import BatchEncoding
from FastChemTokenizer import FastChemTokenizerSelfies
import numpy as np
from collections import Counter
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

# ========================
# UTILS: SELFIES -> SMILES -> VALIDITY & LIPINSKI
# ========================

def selfies_to_smiles(selfies_str: str) -> str | None:
    """Convert SELFIES string to SMILES, handling tokenizer artifacts."""
    try:
        clean_selfies = selfies_str.replace(" ", "")
        return sf.decoder(clean_selfies)
    except Exception:
        return None

def is_valid_smiles(smiles: str) -> bool:
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return False
    return Chem.MolFromSmiles(smiles.strip()) is not None


# ==========================
#  Reward Components
# ==========================
def compute_biological_diversity_score(mol) -> float:
    """Reward molecules with diverse CHONP atoms, normalized to [0,1]."""
    if mol is None:
        return 0.0
    try:
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_counts = Counter(atoms)
        bio_elements = {"C", "H", "O", "N", "P"}
        present_bio_elements = set(atoms) & bio_elements

        if len(present_bio_elements) < 2:
            return 0.0

        base_score = 0.3
        diversity_bonus = (len(present_bio_elements) - 2) / 3 * 0.4

        total_bio_atoms = sum(atom_counts.get(e, 0) for e in present_bio_elements)
        if total_bio_atoms > 0:
            bio_probs = [atom_counts.get(e, 0) / total_bio_atoms for e in present_bio_elements]
            if len(bio_probs) > 1:
                entropy = -sum(p * np.log2(p) for p in bio_probs if p > 0)
                max_entropy = np.log2(len(bio_probs))
                entropy_bonus = (entropy / max_entropy) * 0.3
            else:
                entropy_bonus = 0.0
        else:
            entropy_bonus = 0.0

        return min(1.0, base_score + diversity_bonus + entropy_bonus)
    except Exception:
        return 0.0


def compute_charge_neutrality_score(mol) -> float:
    """Reward if molecule is globally neutral (formal charge = 0)."""
    if mol is None:
        return 0.0
    try:
        return 1.0 if Chem.rdmolops.GetFormalCharge(mol) == 0 else 0.0
    except Exception:
        return 0.0


def compute_local_charge_penalty(mol) -> float:
    """
    Penalize carbocations/anions.
    Returns 1.0 if no charged atoms, decreases with fraction charged.
    """
    if mol is None:
        return 0.0
    try:
        charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        if not charges:
            return 1.0
        charged_atoms = sum(1 for c in charges if c != 0)
        total_atoms = len(charges)
        return max(0.0, 1.0 - (charged_atoms / total_atoms))
    except Exception:
        return 0.0


def compute_enhanced_lipinski_reward(mol) -> float:
    """Soft Lipinski scoring with partial credit."""
    if mol is None:
        return 0.0
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        scores = []

        # MW
        if 250 <= mw <= 500: scores.append(1.0)
        elif 150 <= mw < 250: scores.append(0.5)
        elif 500 < mw <= 600: scores.append(0.7)
        else: scores.append(0.0)

        # LogP
        if -1 <= logp <= 5: scores.append(1.0)
        elif -2 <= logp < -1 or 5 < logp <= 6: scores.append(0.5)
        else: scores.append(0.0)

        # Donors
        scores.append(1.0 if hbd <= 5 else max(0.0, 1.0 - 0.2 * (hbd - 5)))
        # Acceptors
        scores.append(1.0 if hba <= 10 else max(0.0, 1.0 - 0.1 * (hba - 10)))

        return sum(scores) / len(scores)
    except Exception:
        return 0.0


def compute_structural_complexity_reward(mol) -> float:
    """Reward moderate complexity: 1–3 rings and some flexibility."""
    if mol is None:
        return 0.0
    try:
        ring_count = rdMolDescriptors.CalcNumRings(mol)
        if 1 <= ring_count <= 3: ring_score = 1.0
        elif ring_count == 0: ring_score = 0.3
        elif ring_count <= 5: ring_score = 0.7
        else: ring_score = 0.1

        rot_bonds = Descriptors.NumRotatableBonds(mol)
        if 2 <= rot_bonds <= 8: flex_score = 1.0
        elif rot_bonds <= 12: flex_score = 0.7
        elif rot_bonds in (0, 1): flex_score = 0.5
        else: flex_score = 0.2

        return (ring_score + flex_score) / 2
    except Exception:
        return 0.0


# ==========================
#  Unified Reward
# ==========================
def compute_comprehensive_reward(selfies_str: str) -> dict[str, float]:
    smiles = selfies_to_smiles(selfies_str)
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    rewards = {
        "validity": 1.0 if mol is not None else 0.0,
        "biological_diversity": compute_biological_diversity_score(mol),
        "charge_neutrality": compute_charge_neutrality_score(mol),
        "local_charge_penalty": compute_local_charge_penalty(mol),
        "lipinski": compute_enhanced_lipinski_reward(mol),
        "structural_complexity": compute_structural_complexity_reward(mol),
    }

    if rewards["validity"] == 0:
        rewards["total"] = 0.0
    else:
        weights = {
            "validity": 1.0,
            "biological_diversity": 2.0,
            "charge_neutrality": 1.5,
            "local_charge_penalty": 1.0,
            "lipinski": 1.0,
            "structural_complexity": 0.5,
        }
        weighted_sum = sum(rewards[k] * weights[k] for k in weights)
        rewards["total"] = weighted_sum / sum(weights.values())

    return rewards

def compute_lipinski_reward(mol) -> float:
    if mol is None:
        return 0.0
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rules = [250 < mw <= 500, logp <= 5, hbd <= 5, hba <= 10]   # we dont want too small of fragments
        return sum(rules) / 4.0
    except:
        return 0.0

def selfies_to_lipinski_reward(selfies_str: str) -> float:
    """Convert SELFIES to SMILES, then compute Lipinski reward."""
    smiles = selfies_to_smiles(selfies_str)
    if smiles is None:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    return compute_lipinski_reward(mol)

class MTPHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_future_tokens: int = 3):
        super().__init__()
        self.num_future_tokens = num_future_tokens
        self.vocab_size = vocab_size
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_future_tokens)
        ])
        self.position_embeddings = nn.Embedding(num_future_tokens, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        outputs = {}
        for i in range(self.num_future_tokens):
            pos_emb = self.position_embeddings(torch.tensor(i, device=hidden_states.device))
            enhanced_hidden = self.layer_norm(hidden_states + pos_emb)
            logits = self.prediction_heads[i](enhanced_hidden)
            outputs[f'logits_t{i+1}'] = logits
        return outputs


class HorizonLoss(nn.Module):
    def __init__(self, num_future_tokens: int = 3, horizon_weights: Optional[List[float]] = None):
        super().__init__()
        self.num_future_tokens = num_future_tokens
        if horizon_weights is None:
            self.horizon_weights = [0.9 ** i for i in range(num_future_tokens)]
        else:
            self.horizon_weights = horizon_weights
        self.log_weights = nn.Parameter(torch.log(torch.tensor(self.horizon_weights)))
        
    def forward(self, mtp_outputs: Dict[str, torch.Tensor], 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        weights = F.softmax(self.log_weights, dim=0)
        total_loss = 0.0
        horizon_losses = {}
        for i in range(self.num_future_tokens):
            logits_key = f'logits_t{i+1}'
            if logits_key not in mtp_outputs:
                continue
            logits = mtp_outputs[logits_key]
            shift = i + 1
            if seq_len <= shift:
                continue
            shifted_logits = logits[:, :-shift, :].contiguous()
            shifted_targets = input_ids[:, shift:].contiguous()
            if attention_mask is not None:
                shifted_mask = attention_mask[:, shift:].contiguous()
                mask_expanded = shifted_mask.view(-1)
                valid_indices = mask_expanded == 1
                if valid_indices.sum() == 0:
                    continue
                flat_logits = shifted_logits.view(-1, logits.size(-1))[valid_indices]
                flat_targets = shifted_targets.view(-1)[valid_indices]
            else:
                flat_logits = shifted_logits.view(-1, logits.size(-1))
                flat_targets = shifted_targets.view(-1)
            horizon_loss = F.cross_entropy(flat_logits, flat_targets, reduction='mean')
            horizon_losses[f'horizon_loss_t{i+1}'] = horizon_loss
            total_loss += weights[i] * horizon_loss
        return {'loss': total_loss, 'horizon_weights': weights, **horizon_losses}


class ChemQ3MTP(Qwen3ForCausalLM):
    def __init__(self, config, num_future_tokens: int = 3):
        super().__init__(config)
        self.mtp_head = MTPHead(config.hidden_size, config.vocab_size, num_future_tokens)
        self.horizon_loss = HorizonLoss(num_future_tokens=num_future_tokens)
        self.use_mtp_training = True
        self.post_init()
        self.entropy_controller = EnhancedEntropyController(
            min_entropy=0.5,
            max_entropy=3.0,
            target_entropy=1.5,
            adaptation_rate=0.01,
            )


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # ⬅️ ADDED
        **kwargs
    ):
        # Default mask if not provided
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        # Respect caller settings, only set defaults if missing
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("return_dict", True)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # ⬅️ We handle loss ourselves, don't let base class do it
            **kwargs
        )

        hidden_states = outputs.hidden_states[-1]
        lm_logits = outputs.logits
        loss = None

        if self.training and self.use_mtp_training and labels is not None:  # ✅ labels, not kwargs
            mtp_outputs = self.mtp_head(hidden_states)
            horizon_loss_dict = self.horizon_loss(mtp_outputs, input_ids, attention_mask)

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()  # ✅ labels, not kwargs["labels"]

            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
                loss_mask = shift_mask.view(-1) == 1
                if loss_mask.sum() == 0:
                    causal_lm_loss = torch.tensor(0.0, device=lm_logits.device)
                else:
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))[loss_mask]
                    flat_labels = shift_labels.view(-1)[loss_mask]
                    causal_lm_loss = F.cross_entropy(flat_logits, flat_labels, reduction='mean')
            else:
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                causal_lm_loss = F.cross_entropy(flat_logits, flat_labels, reduction='mean')

            loss = 0.7 * horizon_loss_dict['loss'] + 0.3 * causal_lm_loss

        elif labels is not None:  # ✅ labels, not kwargs.get("labels")
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()  # ✅ labels, not kwargs["labels"]
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def set_mtp_training(self, use_mtp: bool):
        self.use_mtp_training = use_mtp

    # ================
    # RL SAMPLING + PPO
    # ================

    def generate_with_logprobs(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        return_probs: bool = True,
        tokenizer=None,   # allow passing explicitly
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self.eval()
        device = input_ids.device

        # Normalize shapes: allow [L], [1,L], [B,L], [B,1,L]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [L] -> [1,L]
        if input_ids.dim() == 3 and input_ids.size(1) == 1:
            input_ids = input_ids.squeeze(1)    # [B,1,L] -> [B,L]
        assert input_ids.dim() == 2, f"input_ids must be 2-D, got {input_ids.shape}"

        batch_size, seq_len = input_ids.shape
        current_input = input_ids

        generated_tokens, generated_logprobs, generated_probs = [], [], []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self(current_input, use_cache=False)
                logits = outputs.logits[:, -1, :] / temperature

                # Top-k
                if top_k is not None:
                    values, indices = torch.topk(logits, k=top_k)
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, indices, values)

                # Top-p
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    logits[mask.scatter(1, sorted_indices, mask)] = float("-inf")

                probs = F.softmax(logits, dim=-1)

                if do_sample:
                    dist = Categorical(probs)
                    next_token = dist.sample()
                    log_p = dist.log_prob(next_token)
                else:
                    next_token = torch.argmax(probs, dim=-1)
                    log_p = torch.log(torch.gather(probs, 1, next_token.unsqueeze(1))).squeeze(1)

                generated_tokens.append(next_token.unsqueeze(1))
                generated_logprobs.append(log_p.unsqueeze(1))
                if return_probs:
                    generated_probs.append(probs.unsqueeze(1))

                current_input = torch.cat([current_input, next_token.unsqueeze(1)], dim=1)

        generated_tokens = torch.cat(generated_tokens, dim=1)      # [B, T]
        generated_logprobs = torch.cat(generated_logprobs, dim=1)  # [B, T]
        generated_probs = torch.cat(generated_probs, dim=1) if return_probs else None

        # Use passed tokenizer, fallback to self.tokenizer
        tok = tokenizer if tokenizer is not None else getattr(self, "tokenizer", None)
        if tok is None:
            raise ValueError("Tokenizer must be provided to decode generated tokens.")

        decoded_list = [
            tok.decode(tok_ids, skip_special_tokens=True)
            for tok_ids in generated_tokens
        ]
        return decoded_list, generated_logprobs, generated_tokens, generated_probs


    def ppo_step(
        self,
        input_ids: torch.LongTensor,
        old_log_probs: torch.Tensor,
        old_action_probs: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        validity_weight: float = 1.0,
        lipinski_weight: float = 1.0,
        entropy_weight: float = 0.01,
        clip_epsilon: float = 0.2,
        baseline: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # =========================
        #  PPO-KL BODY  (drop-in)
        # =========================
        self.train()
        self.set_mtp_training(False)
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = tokenizer

        # Ensure entropy controller exists
        if not hasattr(self, 'entropy_controller'):
            # if you want different defaults, set them when constructing model instead
            self.entropy_controller = EnhancedEntropyController(
                min_entropy=0.5,
                max_entropy=3.0,
                target_entropy=1.5,
                adaptation_rate=0.01
            )

        # --- roll-out ---
        selfies_list, new_log_probs, token_ids, new_action_probs = self.generate_with_logprobs(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            return_probs=True,
            tokenizer=getattr(self, "tokenizer", None),
        )

        # --- rewards: use unified reward function ---
        validity_vals = []
        lipinski_vals = []
        total_rewards = []

        for s in selfies_list:
            r = compute_comprehensive_reward(s)
            # r contains keys: validity, biological_diversity, charge_neutrality,
            # local_charge_penalty, lipinski, structural_complexity, total
            validity_vals.append(r.get('validity', 0.0))
            lipinski_vals.append(r.get('lipinski', 0.0))
            total_rewards.append(r.get('total', 0.0))

        device = new_log_probs.device
        rewards = torch.tensor(total_rewards, dtype=torch.float32, device=device)

        validity_rewards = torch.tensor(validity_vals, dtype=torch.float32, device=device)
        lipinski_rewards = torch.tensor(lipinski_vals, dtype=torch.float32, device=device)

        if baseline is not None:
            rewards = rewards - baseline

        # --- probability ratio ---
        old_probs = torch.gather(old_action_probs, 2, token_ids.unsqueeze(2)).squeeze(2).clamp_min(1e-8)
        new_probs = torch.gather(new_action_probs, 2, token_ids.unsqueeze(2)).squeeze(2).clamp_min(1e-8)
        log_ratio = new_log_probs - old_log_probs
        total_ratio = torch.exp(log_ratio.sum(dim=1))

        # --- adaptive KL controller (singleton) ---
        if not hasattr(self, 'kl_controller'):
            self.kl_controller = AdaptiveKLController()
        kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=1)
        beta = self.kl_controller.update(kl.mean().item())

        # --- PPO-KL loss ---
        surr1 = total_ratio * rewards
        surr2 = torch.clamp(total_ratio, 1 - clip_epsilon, 1 + clip_epsilon) * rewards
        ppo_loss = -torch.min(surr1, surr2).mean()
        kl_penalty = beta * kl.mean()
        total_policy_loss = ppo_loss + kl_penalty

        # --- entropy bonus (adaptive) ---
        # compute token-level entropy averaged across batch/time
        # new_action_probs has shape [B, T, V]; we compute entropy per-step then mean.
        # If new_action_probs is already [B, T, V], this works; otherwise adjust accordingly.
        with torch.no_grad():
            # avoid NaNs
            _probs = new_action_probs.clamp_min(1e-12)
            per_step_entropy = -(_probs * torch.log(_probs)).sum(dim=-1)  # [B, T]
            entropy = per_step_entropy.mean()  # scalar tensor

        # dynamically adjust weight using controller (returns a float)
        adaptive_entropy_weight = self.entropy_controller.update_entropy_weight(entropy.item())

        entropy_bonus = adaptive_entropy_weight * entropy
        total_loss = total_policy_loss - entropy_bonus

        # regularization (optional) - keep your small L2 reg if you like
        reg_loss = 1e-7 * sum(p.pow(2).sum() for p in self.parameters())
        total_loss = total_loss + reg_loss

        # prepare return (detach tensors where relevant)
        return {
            'loss': total_loss,
            'ppo_loss': ppo_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'kl_coef': beta,
            'entropy': entropy.item(),
            'entropy_weight': adaptive_entropy_weight,
            'validity_rate': validity_rewards.mean().item(),
            'lipinski_score': lipinski_rewards.mean().item(),
            'avg_reward': rewards.mean().item(),
            'generated_selfies': selfies_list,
            'generated_smiles': [selfies_to_smiles(s) for s in selfies_list],
            'new_log_probs': new_log_probs.detach(),
            'new_action_probs': new_action_probs.detach(),
        }



# ========================
# CURRICULUM LEARNING MANAGER
# ========================

class CurriculumManager:
    def __init__(self, start_len=10, max_len=80, step_increase=5, steps_per_level=200):
        self.current_max_len = start_len
        self.max_len = max_len
        self.step_increase = step_increase
        self.steps_per_level = steps_per_level
        self.step_counter = 0

    def get_max_new_tokens(self):
        return self.current_max_len

    def step(self):
        self.step_counter += 1
        if self.step_counter % self.steps_per_level == 0 and self.current_max_len < self.max_len:
            self.current_max_len = min(self.current_max_len + self.step_increase, self.max_len)
            print(f"📈 Curriculum Update: max_new_tokens = {self.current_max_len}")
        return self.current_max_len

class AdaptiveKLController:
    """
    Increases or decreases β so that E[KL] stays ≈ target_kl.
    """
    def __init__(self, init_kl_coef: float = 0.1, target_kl: float = 0.01,
                 kl_horizon: int = 1000, increase_rate: float = 1.5, decrease_rate: float = 0.8):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.kl_horizon  = kl_horizon
        self.inc = increase_rate
        self.dec = decrease_rate
        self.buffer = []

    def update(self, kl: float):
        self.buffer.append(kl)
        if len(self.buffer) >= self.kl_horizon:
            avg_kl = sum(self.buffer) / len(self.buffer)
            self.buffer.clear()
            if avg_kl > self.target_kl * 1.5:
                self.kl_coef *= self.inc
            elif avg_kl < self.target_kl * 0.5:
                self.kl_coef *= self.dec
        return self.kl_coef
    

class EnhancedEntropyController:
    """
    More sophisticated entropy control with dynamic targets and temperature scheduling.
    """
    def __init__(self, min_entropy: float = 0.5, max_entropy: float = 3.0,
                 target_entropy: float = 1.5, adaptation_rate: float = 0.01):
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.target_entropy = target_entropy
        self.adaptation_rate = adaptation_rate
        self.entropy_history = []
        self.entropy_weight = 0.01  # Starting weight
        
    def update_entropy_weight(self, current_entropy: float) -> float:
        """
        Dynamically adjust entropy weight based on current entropy levels.
        """
        self.entropy_history.append(current_entropy)
        
        # Keep rolling window
        if len(self.entropy_history) > 100:
            self.entropy_history = self.entropy_history[-100:]
            
        if len(self.entropy_history) >= 10:
            avg_entropy = np.mean(self.entropy_history[-10:])
            
            # If entropy too low, increase weight to encourage exploration
            if avg_entropy < self.target_entropy * 0.8:
                self.entropy_weight = min(0.05, self.entropy_weight * 1.1)
            # If entropy too high, decrease weight
            elif avg_entropy > self.target_entropy * 1.2:
                self.entropy_weight = max(0.001, self.entropy_weight * 0.95)
                
        return self.entropy_weight
    
    def compute_entropy_reward(self, entropy: float) -> float:
        """
        Reward function for entropy - prefer target range.
        """
        if self.min_entropy <= entropy <= self.max_entropy:
            # Gaussian reward centered at target
            distance = abs(entropy - self.target_entropy)
            max_distance = max(self.target_entropy - self.min_entropy, 
                             self.max_entropy - self.target_entropy)
            return np.exp(-(distance / max_distance) ** 2)
        else:
            return 0.1  # Small penalty for being outside range