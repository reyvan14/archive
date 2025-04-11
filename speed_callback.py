# speed_callback.py
import time
import wandb
from transformers import TrainerCallback

class SpeedTrackingCallback(TrainerCallback):
    def __init__(self):
        self.last_time = None

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if self.last_time is not None:
            step_time = now - self.last_time
            steps_per_second = 1 / step_time if step_time > 0 else 0
            wandb.log({
                "train/step_time": step_time,
                "train/steps_per_second": steps_per_second
            }, step=state.global_step)
        self.last_time = now
