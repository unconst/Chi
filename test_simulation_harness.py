import unittest

from simulation_harness import MinerProfile, run_simulation


class SimulationHarnessTests(unittest.TestCase):
    def test_simulation_is_deterministic(self):
        miners = [
            MinerProfile(uid=1, name="honest", base_valid_rate=0.92, base_elegance=0.82),
            MinerProfile(uid=2, name="noisy", base_valid_rate=0.70, base_elegance=0.65),
            MinerProfile(uid=3, name="adversarial", base_valid_rate=0.45, base_elegance=0.40),
        ]
        w1 = run_simulation(
            seed=42,
            netuid=1,
            start_block=1000,
            rounds=40,
            batch_size=6,
            ema_alpha=0.35,
            miners=miners,
        )
        w2 = run_simulation(
            seed=42,
            netuid=1,
            start_block=1000,
            rounds=40,
            batch_size=6,
            ema_alpha=0.35,
            miners=miners,
        )
        self.assertEqual(w1, w2)

    def test_honest_miner_ranks_first(self):
        miners = [
            MinerProfile(uid=11, name="honest", base_valid_rate=0.93, base_elegance=0.84),
            MinerProfile(uid=12, name="noisy", base_valid_rate=0.68, base_elegance=0.63),
            MinerProfile(uid=13, name="adversarial", base_valid_rate=0.40, base_elegance=0.50),
        ]
        weights = run_simulation(
            seed=7,
            netuid=1,
            start_block=2000,
            rounds=60,
            batch_size=6,
            ema_alpha=0.30,
            miners=miners,
        )
        ordered = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        self.assertEqual(ordered[0][0], 11)
        self.assertGreater(weights[11], weights[12])
        self.assertGreater(weights[12], weights[13])
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
