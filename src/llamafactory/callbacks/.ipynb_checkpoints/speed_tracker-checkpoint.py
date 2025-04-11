

import time
from transformers import TrainerCallback


class SpeedTrackerCallback(TrainerCallback):
    def __init__(self, output_path="tpm_log.txt", pad_token_id=0):
        self.start_time = time.time()
        self.total_tokens = 0
        self.output_path = output_path
        self.pad_token_id = pad_token_id

        with open(self.output_path, "w") as f:
            f.write("Step\tTotalTokens\tTPM\n")

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        inputs = kwargs.get("inputs")
        if inputs is not None and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            valid_tokens = (input_ids != self.pad_token_id).sum().item()
            self.total_tokens += valid_tokens

        elapsed = time.time() - self.start_time
        if elapsed > 0:
            tpm = (self.total_tokens / elapsed) * 60
            with open(self.output_path, "a") as f:
                f.write(f"{state.global_step}\t{self.total_tokens}\t{tpm:.2f}\n")

        return control
