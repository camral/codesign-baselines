"""Budget checks around unchanged SB3 PPO minibatch updates."""

from stable_baselines3 import PPO


class TrainingDeadlineReached(Exception):
    pass


class TimedPPO(PPO):
    def __init__(self, *args, budget=None, progress_hook=None, **kwargs):
        self.budget = budget
        self.progress_hook = progress_hook
        self.completed_ppo_iterations = 0
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['budget', 'progress_hook']

    def train(self):
        original_get = self.rollout_buffer.get

        def check():
            if self.budget is not None and self.budget.expired:
                raise TrainingDeadlineReached()
            if self.progress_hook is not None:
                self.progress_hook()
            if self.budget is not None and self.budget.expired:
                raise TrainingDeadlineReached()

        def batches(batch_size=None):
            check()
            for batch in original_get(batch_size):
                check()
                yield batch

        self.rollout_buffer.get = batches
        try:
            super().train()
            self.completed_ppo_iterations += 1
        finally:
            self.rollout_buffer.get = original_get
