from unittest import TestCase
from PlanningAgent import ReplayMemory
import numpy as np


class TestReplayMemory(TestCase):

    def fake_transition(self, val_s=0, val_a=0, val_r=0, val_s_=0, val_done=False):
        tpl = [np.full(self.mem._data_dtype[0][2], val_s),
               np.full(self.mem._data_dtype[1][2], val_a),
               np.full(self.mem._data_dtype[2][2], val_r),
               np.full(self.mem._data_dtype[3][2], val_s_),
               np.full(self.mem._data_dtype[4][2], val_done)]
        tpl.extend([np.zeros(dtype[2], dtype=dtype[1]) for dtype in self.mem._data_dtype[5:]])
        return tuple(tpl)

    def fake_trajectory(self, n_transitions, field_vals=0, final_terminal=True):
        traj = [self.fake_transition(field_vals, field_vals, field_vals, field_vals) for _ in range(n_transitions - 1)]
        traj.append(self.fake_transition(field_vals, field_vals, field_vals, field_vals, val_done=final_terminal))
        return traj

    def setUp(self) -> None:
        n_max_elements = 10
        s_obs = ()
        s_act = ()
        s_reward = ()
        custom_fields = None
        self.mem = ReplayMemory(n_max_elements, s_obs, s_act, s_reward, custom_fields)

    def test_add(self):
        # add first transition
        self.mem.clear()
        trans = self.fake_transition(val_s = 10, val_s_= 20)
        self.mem.add(*trans)
        self.assertEqual(self.mem.added_transitions, 0)

        # s_ from last transition differs from s in this one, that's invalid
        invalid_trans = self.fake_transition(val_s = 10, val_s_= 20)
        self.assertRaises(ValueError, self.mem.add, *invalid_trans)

        # add terminal transition
        terminal_trans = self.fake_transition(val_s = 20, val_s_= 20, val_done=True)
        self.mem.add(*terminal_trans)
        self.assertEqual(self.mem.added_transitions, 2)

        # too long trajectory
        self.mem.clear()
        for i in range(self.mem.n_max_elements):
            trans = self.fake_transition(val_s = 1, val_s_= 1)
            self.mem.add(*trans)
        terminal_trans = self.fake_transition(val_s = 1, val_s_= 20, val_done=True)
        self.assertRaises(RuntimeError, self.mem.add, *terminal_trans)

        # add more than one transition
        self.mem.clear()
        for trans in self.fake_trajectory(2):
            self.mem.add(*trans)
        for trans in self.fake_trajectory(2):
            self.mem.add(*trans)
        self.assertEqual(self.mem.added_transitions, 4)
        self.assertEqual(self.mem.valid_transitions, 4)
        # overwrite the two added transitions with one large one
        for trans in self.fake_trajectory(7, final_terminal=False):
            self.mem.add(*trans)
        self.mem.add(*self.fake_transition(0, 10, 10, 10, True))
        self.assertEqual(self.mem.added_transitions, 12)
        self.assertEqual(self.mem.valid_transitions, 8)


    def test_sample_transitions(self):
        self.mem.clear()
        # add two trajectories
        self.mem.add(*self.fake_transition(0, 0, 0, 1, False))
        self.mem.add(*self.fake_transition(1, 0, 0, 2, False))
        self.mem.add(*self.fake_transition(2, 0, 0, 3, True))
        self.mem.add(*self.fake_transition(0, 0, 0, 10, False))
        self.mem.add(*self.fake_transition(10, 0, 0, 20, False))
        self.mem.add(*self.fake_transition(20, 0, 0, 30, True))

        # sample deterministically and check for consistency
        transitions = self.mem.sample_transitions(4, debug=True)
        self.assertEqual(transitions[0]['s'], 0)
        self.assertEqual(transitions[0]['s_'], 1)
        self.assertEqual(transitions[1]['s'], 1)
        self.assertEqual(transitions[1]['s_'], 2)
        self.assertEqual(transitions[2]['s'], 2)
        self.assertEqual(transitions[2]['s_'], 3)
        self.assertEqual(transitions[3]['s'], 0)
        self.assertEqual(transitions[3]['s_'], 10)

    def test_sample_trajectories(self):
        self.mem.clear()
        # add two trajectories
        self.mem.add(*self.fake_transition(0, 0, 0, 1, False))
        self.mem.add(*self.fake_transition(1, 0, 0, 2, False))
        self.mem.add(*self.fake_transition(2, 0, 0, 3, True))
        self.mem.add(*self.fake_transition(0, 0, 0, 10, False))
        self.mem.add(*self.fake_transition(10, 0, 0, 20, False))
        self.mem.add(*self.fake_transition(20, 0, 0, 30, True))

        # sample deterministically and check for consistency
        trajectories = self.mem.sample_trajectories(2, 2, debug=True)
        self.assertEqual(trajectories.shape, (2, 2))
        self.assertEqual(trajectories[0, 0]['s'], 0)
        self.assertEqual(trajectories[0, 0]['s_'], 1)
        self.assertEqual(trajectories[0, 1]['s'], 1)
        self.assertEqual(trajectories[0, 1]['s_'], 2)
        self.assertEqual(trajectories[1, 0]['s'], 0)
        self.assertEqual(trajectories[1, 0]['s_'], 10)
        self.assertEqual(trajectories[1, 1]['s'], 10)
        self.assertEqual(trajectories[1, 1]['s_'], 20)
