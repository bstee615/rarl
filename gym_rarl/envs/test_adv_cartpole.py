import unittest

import numpy as np

from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv


class MyTestCase(unittest.TestCase):

    def setup_method(self, _):
        self.control_env = AdversarialCartPoleEnv()
        self.adv_env = AdversarialCartPoleEnv()

        self.control_env.seed(0)
        self.adv_env.seed(0)
        self.control_env.reset()
        self.adv_env.reset()

    def teardown_method(self, _):
        self.control_env.close()
        self.adv_env.close()

    def test_reset(self):
        self.control_env.seed(0)
        self.adv_env.seed(0)
        control_initial_state = self.control_env.reset()
        adv_initial_state = self.adv_env.reset()

        np.testing.assert_almost_equal(control_initial_state, adv_initial_state)
        np.testing.assert_almost_equal(control_initial_state,
                                       np.array([-0.04456399, 0.04653909, 0.01326909, -0.02099827]))

    def test_default_step(self):
        np.testing.assert_almost_equal(self.control_env.step(1)[0], self.adv_env.step(1)[0])

    def test_adv_noop(self):
        np.testing.assert_almost_equal(self.control_env.step(1)[0], self.adv_env.step_two_agents(1, 0)[0])

    def test_adv_diff(self):
        np.testing.assert_raises(AssertionError, np.testing.assert_almost_equal,
                                 self.control_env.step_two_agents(1, -1)[0], self.adv_env.step_two_agents(1, 1)[0])


if __name__ == '__main__':
    unittest.main()
