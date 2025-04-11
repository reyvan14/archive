from transformers import Trainer

class TokenLoggingTrainer(Trainer):
    def training_step(self, model, inputs):
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            actual_tokens = input_ids.ne(0).sum().item()
            self.control = self.callback_handler.on_log(
                self.args, self.state, self.control, logs={"actual_tokens": actual_tokens}
            )
        return super().training_step(model, inputs)
