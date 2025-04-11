import os
import time
from transformers import TrainerCallback

class SpeedLoggerCallback(TrainerCallback):
    def __init__(self, output_path="tpm_log.txt", pad_token_id=0):
        self.start_time = time.time()
        self.total_effective_tokens = 0
        self.output_path = output_path
        self.pad_token_id = pad_token_id
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            f.write("Step\tEffectiveTokens\tTPM\n")

    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        # 从日志或 batch 提取有效 token 数（实际逻辑中可替换成实际 batch 计算）
        effective_tokens = logs.get("effective_token_count", 4096 * args.per_device_train_batch_size)

        self.total_effective_tokens += effective_tokens
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            tpm = (self.total_effective_tokens / elapsed) * 60
            with open(self.output_path, "a") as f:
                f.write(f"{state.global_step}\t{self.total_effective_tokens}\t{tpm:.2f}\n")
        return control
