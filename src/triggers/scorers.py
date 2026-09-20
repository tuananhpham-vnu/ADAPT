"""Constraint helpers shared by AgentPoison optimization and tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def coherence_sampling_probabilities(losses: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if losses.ndim != 1:
        raise ValueError("losses must be one-dimensional")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    
    return torch.softmax(-losses / float(temperature), dim=0)


def sample_coherence_candidates(
    losses: torch.Tensor,
    sample_size: int, 
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 1 <= sample_size <= losses.numel():
        raise ValueError("sample_size must be in [1, candidate_count]")
    
    probabilities = coherence_sampling_probabilities(losses, temperature)
    indices = torch.multinomial(probabilities, sample_size, replacement=False, generator=generator)
    
    return indices, probabilities


def geometric_target_probability(token_log_probabilities: torch.Tensor) -> torch.Tensor:
    """Geometric mean probability over target tokens (optionally batched)."""
    if token_log_probabilities.ndim == 0:
        raise ValueError("at least one target-token dimension is required")
    
    return token_log_probabilities.mean(dim=-1).exp()


def select_candidate(
    current_loss: float,
    current_target_probability: float,
    candidate_losses: Sequence[float],
    candidate_target_probabilities: Sequence[float],
    threshold: float = 0.8,
) -> int | None:
    """Return an accepted candidate index using the target-feasibility gate."""
    if len(candidate_losses) != len(candidate_target_probabilities):
        raise ValueError("candidate loss/probability arrays must be aligned")
    
    improving = [i for i, value in enumerate(candidate_losses) if value < current_loss]
    improving.sort(key=lambda i: candidate_losses[i])
    feasible = [i for i in improving if candidate_target_probabilities[i] >= threshold]
    
    if feasible:
        return feasible[0]
    
    if current_target_probability >= threshold:
        return None
    
    bootstrap = [
        i for i in improving
        if candidate_target_probabilities[i] > current_target_probability
    ]
    
    return bootstrap[0] if bootstrap else None


@dataclass
class GPT2CoherenceScorer:
    model: object
    tokenizer: object
    device: str = "cuda:0"

    def score(self, queries: Sequence[str], trigger_texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean causal NLL and perplexity for each candidate trigger."""
        losses = []
        self.model.eval()
        
        with torch.no_grad():
            for trigger in trigger_texts:
                per_query = []
                for query in queries:
                    encoded = self.tokenizer(
                        f"{query}\n{trigger}", return_tensors="pt", truncation=True
                    )
                    encoded = {key: value.to(self.device) for key, value in encoded.items()}
                    per_query.append(self.model(**encoded, labels=encoded["input_ids"]).loss.float())
                losses.append(torch.stack(per_query).mean())
                
        nll = torch.stack(losses)
        return nll, nll.exp()


@dataclass
class LlamaTargetScorer:
    model: object
    tokenizer: object
    device: str = "cuda:1"
    target: str = "Finish[I don't know]"

    def _one(self, prompt: str) -> torch.Tensor:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids # dim 1,Q
        target_ids = self.tokenizer(self.target, add_special_tokens=False, return_tensors="pt").input_ids # dim 1,T   
        input_ids = torch.cat((prompt_ids, target_ids), dim=1).to(self.device) # dim 1,Q+T
        
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits[:, :-1].float() # dim 1,Q+T-1,V; Bỏ logits của token cuối cùng vì không có token tiếp theo để dự đoán
            
        start = prompt_ids.shape[1] - 1 # 
        target_logits = logits[:, start:start + target_ids.shape[1]]
        labels = target_ids.to(self.device)
        log_probs = torch.log_softmax(target_logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        
        return geometric_target_probability(log_probs).squeeze(0)

    def score(self, prompts: Sequence[str]) -> torch.Tensor:
        return torch.stack([self._one(prompt) for prompt in prompts]).mean()
