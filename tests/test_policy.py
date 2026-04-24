from proactive_kv_cache.cache import TieredStateBank
from proactive_kv_cache.policy import CostAwareSlackPolicy


def test_policy_returns_ranked_candidates():
    bank = TieredStateBank(max_memory_bytes=4096, min_match_length=3)
    for _ in range(12):
        bank.observe_query((1, 2, 3, 4, 5, 6))
    for _ in range(3):
        bank.observe_query((1, 2, 3, 9, 9, 9))

    policy = CostAwareSlackPolicy(
        min_frequency=0.10,
        benefit_cost_ratio=0.20,
        speculation_penalty_ms=1.0,
        memory_penalty_per_mb=0.1,
        max_admissions_per_idle=2,
        min_prefix_len=3,
        max_prefix_len=24,
    )
    ranked = policy.rank(bank, budget_k=2)
    assert len(ranked) >= 1
    assert ranked[0].score >= ranked[-1].score
    assert len(ranked) <= 2


def test_policy_filters_weak_candidates_under_strict_gate():
    bank = TieredStateBank(max_memory_bytes=4096, min_match_length=3)
    for _ in range(2):
        bank.observe_query((1, 2, 3, 4, 5, 6))

    policy = CostAwareSlackPolicy(min_frequency=0.30, benefit_cost_ratio=2.0)
    ranked = policy.rank(bank, budget_k=4)
    assert ranked == []