from proactive_kv_cache.energy import EnergySnapshot, NvidiaEnergyMeter
from proactive_kv_cache.utility_admission import OnlineUtilityEstimator, prefix_length_bucket


def test_energy_delta_from_total_energy_counter():
    before = EnergySnapshot(timestamp_s=10.0, gpu_energy_mj=1000, gpu_power_w=50.0, source='nvml')
    after = EnergySnapshot(timestamp_s=12.0, gpu_energy_mj=3400, gpu_power_w=55.0, source='nvml')

    delta = NvidiaEnergyMeter.delta(before, after)

    assert delta['gpu_energy_j'] == 2.4
    assert delta['avg_power_w_from_energy'] == 1.2
    assert delta['energy_source'] == 'nvml'


def test_prefix_length_buckets_are_stable():
    assert prefix_length_bucket(1) == '0-64'
    assert prefix_length_bucket(128) == '65-128'
    assert prefix_length_bucket(257) == '257-512'
    assert prefix_length_bucket(5000) == '4097+'


def test_online_utility_estimator_admits_positive_net_utility():
    estimator = OnlineUtilityEstimator(default_full_ms_per_token=0.5, default_reuse_overhead_ms=2.0)

    decision = estimator.decide(prefix_tokens=16, suffix_tokens=4, min_net_saved_ms=0.0)

    assert decision.admit is True
    assert decision.net_utility_ms > 0.0
    assert decision.bucket == '0-64'


def test_online_utility_estimator_bypasses_negative_utility():
    estimator = OnlineUtilityEstimator(default_full_ms_per_token=0.01, default_reuse_overhead_ms=5.0)

    decision = estimator.decide(prefix_tokens=16, suffix_tokens=4, min_net_saved_ms=0.0)

    assert decision.admit is False
    assert decision.reason == 'negative_net_utility'
    snapshot = estimator.snapshot()
    assert snapshot['utility_bucket_0_64_negative_bypass'] == 1
