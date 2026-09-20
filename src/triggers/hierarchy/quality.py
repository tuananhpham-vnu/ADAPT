"""Independent semantic proxy plus protected-number/negation checks."""
from .text import protected_tokens


class MeaningGuard:
    def __init__(self, encoder, threshold=.85):
        if not 0 <= threshold <= 1:
            raise ValueError("Semantic threshold must be in [0, 1]")
        self.encoder, self.threshold = encoder, threshold

    def check(self, originals, altered):
        similarities = (self.encoder.encode(originals) * self.encoder.encode(altered)).sum(1).clamp(-1, 1).tolist()
        unchanged = [protected_tokens(a) == protected_tokens(b) for a, b in zip(originals, altered)]
        return [{"semantic_similarity": s, "protected_tokens_unchanged": ok,
                 "meaning_proxy_pass": bool(ok and s >= self.threshold)} for s, ok in zip(similarities, unchanged)]
