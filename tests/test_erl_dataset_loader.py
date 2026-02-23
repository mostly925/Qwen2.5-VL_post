import unittest

from run_qwen_rl import compute_accuracy_reward


class TestERLRewardNormalization(unittest.TestCase):
    def test_synonym_and_order_are_normalized(self) -> None:
        generated = "<answer>跳起, 巴黎铁塔</answer>"
        target = "埃菲尔铁塔，跳跃"
        reward = compute_accuracy_reward(generated, target)
        self.assertEqual(reward, 1.0)

    def test_punctuation_difference_does_not_hurt(self) -> None:
        generated = "埃菲尔铁塔。跳跃！"
        target = "埃菲尔铁塔，跳跃"
        reward = compute_accuracy_reward(generated, target)
        self.assertEqual(reward, 1.0)

    def test_numeric_tolerance_gets_high_reward(self) -> None:
        generated = "总数是99.5"
        target = "100"
        reward = compute_accuracy_reward(generated, target)
        self.assertGreaterEqual(reward, 0.75)

    def test_wrong_answer_gets_low_reward(self) -> None:
        generated = "伦敦塔桥，站立"
        target = "埃菲尔铁塔，跳跃"
        reward = compute_accuracy_reward(generated, target)
        self.assertLess(reward, 0.4)


if __name__ == "__main__":
    unittest.main()
