# 环境信息

## 模型
- Qwen 2.5-32B

## GPU型号
- H20-NVLink

```
mmlu得分：
        Average: 83.20                                                                                                                                              
           STEM: 82.77
Social Sciences: 89.67
     Humanities: 78.49
          Other: 84.30
```

# 部署指令

```bash
pip install flash-attn --no-build-isolation
pip install vllm==0.8.1
```


# 训练命令（单卡）

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

---

## 默认 YAML 配置

```yaml
### model
model_name_or_path: ../Qwen/Qwen2.5-32B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: d1
template: qwen
cutoff_len: 4096

max_samples: 1000000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/Qwen2.5-32B-Instructs/lora/sft_fast
logging_steps: 1
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: wandb
run_name: speed_test_fast2

### train
per_device_train_batch_size: 12
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

- 日志记录([output1.log](output1.log))
```
***** train metrics *****
  epoch                    =         2.0
  total_flos               = 575676608GF
  train_loss               =       0.463
  train_runtime            =  1:33:55.93
  train_samples_per_second =       1.345
  train_steps_per_second   =       0.028
```

```yaml
num_train_epochs: 1.0
```

```
***** train metrics *****
  epoch                    =         1.0
  total_flos               = 288137054GF
  train_loss               =      0.5463
  train_runtime            =  0:47:05.43
  train_samples_per_second =       1.341
  train_steps_per_second   =       0.028

```



## 优化后 YAML 配置


```yaml
### 增加以下参数

flash_attn: fa2
group_by_length: true
num_train_epochs: 2.0

```
- 日志记录([output2.log](output2.log))

```
***** train metrics *****
  epoch                    =         2.0
  total_flos               = 383384639GF
  train_loss               =      0.4438
  train_runtime            =  1:03:04.48
  train_samples_per_second =       2.002
  train_steps_per_second   =       0.042
```

```yaml
### 修改批次
num_train_epochs: 1.0
```

```
***** train metrics *****
  epoch                    =         1.0
  total_flos               = 191556914GF
  train_loss               =      0.5122
  train_runtime            =  0:31:36.58
  train_samples_per_second =       1.998
  train_steps_per_second   =       0.042
```


```yaml
### 修改为以下部分

flash_attn: fa2
use_unsloth: true
gradient_checkpointing: true
group_by_length: true
num_train_epochs: 1.0
```

```
  ***** train metrics *****
  epoch                    =         1.0
  total_flos               = 191556914GF
  train_loss               =      0.5122
  train_runtime            =  0:28:30.95
  train_samples_per_second =       2.215
  train_steps_per_second   =       0.046
```

```yaml
### 调整了per_device_train_batch_size和gradient_accumulation_steps
flash_attn: fa2
use_unsloth: true
gradient_checkpointing: true
group_by_length: true
per_device_train_batch_size: 40
gradient_accumulation_steps: 10


```

```
***** train metrics *****
  epoch                    =      0.9474
  total_flos               = 194352414GF
  train_loss               =      1.0467
  train_runtime            =  0:27:43.16
  train_samples_per_second =       2.278
  train_steps_per_second   =       0.005

```


优化后时间提升：**41.13%**
```
mmlu得分：
        Average: 83.29                                                                                                                                                
           STEM: 82.24
Social Sciences: 89.67
     Humanities: 78.92
          Other: 84.55
```



## 优化的代码

```python
#/root/miniconda3/envs/test/lib/python3.10/site-packages/unsloth_zoo/utils.py

__all__ = [
    "Version",
    "_get_dtype",
    "is_main_process",
    "is_distributed",
    "distributed_function",
]

from packaging.version import Version as TrueVersion
import torch

def Version(version):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        return TrueVersion(version)
    except:
        from inspect import getframeinfo, stack
        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Unsloth: Could not get version for `{version}`\n"\
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )
    pass
pass


__DTYPE_MAP = {
    "float32": torch.float32,
    torch.float32: torch.float32,
    "float16": torch.float16,
    torch.float16: torch.float16,
    "bfloat16": torch.bfloat16,
    torch.bfloat16: torch.bfloat16,
}
def _get_dtype(dtype):
    try:
        return __DTYPE_MAP[dtype]
    except:
        if type(dtype) is str:
            try: dtype = eval(f"torch.{dtype.lower()}")
            except: pass
        if type(dtype) is torch.dtype: return dtype
    return None
pass


def is_main_process():
    is_initialized = torch.distributed.is_initialized()
    return (not is_initialized) or (is_initialized and torch.distributed.get_rank() == 0)
pass


def is_distributed():
    return torch.distributed.is_initialized()
pass


def distributed_function(n = 1, function = None, *args, **kwargs):
    result = None  # ✅ 默认初始化，防止未定义
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            object_list = function(*args, **kwargs)
            if n == 1:
                object_list = [object_list]
        else:
            object_list = [None for _ in range(n)]
        torch.distributed.broadcast_object_list(object_list, src = 0, device = "cuda")
        if n == 1:
            result = object_list[0]
        else:
            result = object_list
    else:
        result = function(*args, **kwargs)
    return result

pass
```

```python

# src/llamafactory/hparams/parser.py
def get_train_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> _TRAIN_CLS:
    # 解析标准参数
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)

    # 处理自定义参数，例如 Unsloth 优化
    custom_params = {
        "distributed_training": args.get("distributed_training", None),
        "flash_attention": args.get("flash_attention", None),
        "use_unsloth": args.get("use_unsloth", None),
        "dynamic_batching": args.get("dynamic_batching", None),
        "precision": args.get("precision", None),
        "optimizer": args.get("optimizer", None),
        "pipeline_parallelism": args.get("pipeline_parallelism", None),
        "model_compression": args.get("model_compression", None),
        # 可以继续添加其他自定义的参数...
    }

    # 现在可以根据 custom_params 来设置优化器、并行方式等
    if custom_params["distributed_training"]:
        # 设置分布式训练相关配置
        logger.info("Setting up distributed training with parameters: %s", custom_params["distributed_training"])
    
    if custom_params["flash_attention"]:
        # 启用 FlashAttention 或其他相关优化
        logger.info("Enabling FlashAttention optimization.")
    
    if custom_params["use_unsloth"]:
        # 启用 Unsloth 优化
        logger.info("Enabling Unsloth optimization.")

    # 继续进行后续的标准参数验证和设置
    _set_transformers_logging()

    # 检查参数的有效性
    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args, training_args)

    # 检查 finetuning 参数阶段是否符合预期
    if finetuning_args.stage != "sft":
        if training_args.predict_with_generate:
            raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

        if data_args.neat_packing:
            raise ValueError("`neat_packing` cannot be set as True except SFT.")

        if data_args.train_on_prompt or data_args.mask_history:
            raise ValueError("`train_on_prompt` or `mask_history` cannot be set as True except SFT.")

    # 检查训练和评估相关配置
    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
        raise ValueError("Please launch distributed training with `llamafactory-cli` or `torchrun`.")

    if training_args.deepspeed and training_args.parallel_mode != ParallelMode.DISTRIBUTED:
        raise ValueError("Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify dataset for training.")

    if (training_args.do_eval or training_args.do_predict) and (
        data_args.eval_dataset is None and data_args.val_size < 1e-6
    ):
        raise ValueError("Please specify dataset for evaluation.")

    # 后处理训练参数，设置 `ddp_find_unused_parameters` 等
    if (
        training_args.parallel_mode == ParallelMode.DISTRIBUTED
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning_rank0("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args.ddp_find_unused_parameters = False

    # 处理模型的计算精度
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    # 处理设备映射和最大长度
    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    # 打印日志信息
    logger.info(
        f"Process rank: {training_args.process_index}, "
        f"world size: {training_args.world_size}, device: {training_args.device}, "
        f"distributed training: {training_args.parallel_mode == ParallelMode.DISTRIBUTED}, "
        f"compute dtype: {str(model_args.compute_dtype)}"
    )
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args
```

```python
#src/llamafactory/train/tuner.py
def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    # ✅ 加载训练相关参数（提前）
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    # ✅ 加载 tokenizer（必要：用于获取 pad_token_id）
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    # ✅ 插入自定义的样本速率监控 Callback
    callbacks = callbacks or []
    callbacks.append(SpeedLoggerCallback(
        output_path=os.path.join(training_args.output_dir, "tpm_log.txt"),
        pad_token_id=tokenizer.pad_token_id,
    ))

    # ✅ Ray 分布式处理分支
    ray_args = get_ray_args(args)
    if ray_args.use_ray:
        from ray.train.huggingface.transformers import RayTrainReportCallback
        callbacks.append(RayTrainReportCallback())
        trainer = get_ray_trainer(
            training_function=_training_function,
            train_loop_config={"args": args, "callbacks": callbacks},
            ray_args=ray_args,
        )
        trainer.fit()
    else:
        # ✅ 非 Ray 模式直接执行训练函数
        _training_function(config={"args": args, "callbacks": callbacks})

```

# 训练多卡（以双卡演示）
### 启动命令
```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```


```yaml
deepspeed: examples/deepspeed/ds_z2_config.json
```

```
***** train metrics *****
  epoch                    =      0.9873
  total_flos               = 284805667GF
  train_loss               =      0.6513
  train_runtime            =  0:25:14.32
  train_samples_per_second =       2.502
  train_steps_per_second   =       0.026
```




```yaml
deepspeed: examples/deepspeed/ds_z2_config.json
flash_attn: fa2
use_unsloth: true
gradient_checkpointing: true
group_by_length: true
per_device_train_batch_size: 40
gradient_accumulation_steps: 10

```

```
***** train metrics *****
  epoch                    =      0.8333
  total_flos               = 170395084GF
  train_loss               =      1.1122
  train_runtime            =  0:13:45.14
  train_samples_per_second =       4.592
  train_steps_per_second   =       0.005
```
优化后时间提升：**45.51%**

# VLLM推理部分
## 启动api命令和配置
```bash
llamafactory-cli api examples/inference/llama3_vllm.yaml
```


```yaml
model_name_or_path: ../Qwen/Qwen2.5-32B-Instruct
adapter_name_or_path: saves/Qwen2.5-32B-Instructs/lora/sft_fast

template: qwen
infer_backend: vllm
vllm_enforce_eager: true
```
```
Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14685
Elapsed Time       : 93.16 sec
TPM (tokens/min)   : 9458.30
QPS (req/sec)      : 0.34
Avg Latency        : 20.61 sec
Max Latency        : 37.25 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14786
Elapsed Time       : 94.35 sec
TPM (tokens/min)   : 9403.08
QPS (req/sec)      : 0.34
Avg Latency        : 20.40 sec
Max Latency        : 29.68 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 15181
Elapsed Time       : 96.92 sec
TPM (tokens/min)   : 9398.01
QPS (req/sec)      : 0.33
Avg Latency        : 21.62 sec
Max Latency        : 30.12 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 15244
Elapsed Time       : 95.17 sec
TPM (tokens/min)   : 9610.35
QPS (req/sec)      : 0.34
Avg Latency        : 21.91 sec
Max Latency        : 28.61 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 15244
Elapsed Time       : 95.17 sec
TPM (tokens/min)   : 9610.35
QPS (req/sec)      : 0.34
Avg Latency        : 21.91 sec
Max Latency        : 28.61 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14856
Elapsed Time       : 94.55 sec
TPM (tokens/min)   : 9427.18
QPS (req/sec)      : 0.34
Avg Latency        : 20.12 sec
Max Latency        : 31.50 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14477
Elapsed Time       : 91.51 sec
TPM (tokens/min)   : 9491.72
QPS (req/sec)      : 0.35
Avg Latency        : 19.30 sec
Max Latency        : 28.23 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 15414
Elapsed Time       : 99.16 sec
TPM (tokens/min)   : 9326.40
QPS (req/sec)      : 0.32
Avg Latency        : 22.61 sec
Max Latency        : 34.32 sec
```


### 评测代码：
````python
import time
import requests
import concurrent.futures
import argparse
import os
from datetime import datetime

url = "http://localhost:8008/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 创建输出目录
os.makedirs("outputs", exist_ok=True)

def send_prompt(prompt, max_tokens, index):
    data = {
        "model": "Qwen2.5-32B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }

    start = time.time()
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        latency = time.time() - start
        if response.status_code == 200:
            result = response.json()
            usage = result.get("usage", {})
            tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            content = result["choices"][0]["message"]["content"]

            # 写入结果到 outputs/output_{index}.txt
            filename = f"outputs/output_{index}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)

            return tokens, latency
        else:
            print(f"[{index}] Error {response.status_code}: {response.text}")
            return 0, latency
    except Exception as e:
        print(f"[{index}] Exception: {e}")
        return 0, 0

def benchmark(n_requests, concurrency, prompt, max_tokens):
    print(f"\n🚀 Running {n_requests} requests with concurrency={concurrency} ...")
    start_time = time.time()

    total_tokens = 0
    latencies = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_prompt, prompt, max_tokens, i)
            for i in range(n_requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            tokens, latency = future.result()
            total_tokens += tokens
            if latency > 0:
                latencies.append(latency)

    elapsed = time.time() - start_time
    tpm = total_tokens / elapsed * 60
    qps = n_requests / elapsed
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    print(f"\n📊 Benchmark Summary:")
    print(f"Total Requests     : {n_requests}")
    print(f"Successful Requests: {len(latencies)}")
    print(f"Total Tokens       : {total_tokens}")
    print(f"Elapsed Time       : {elapsed:.2f} sec")
    print(f"TPM (tokens/min)   : {tpm:.2f}")
    print(f"QPS (req/sec)      : {qps:.2f}")
    print(f"Avg Latency        : {avg_latency:.2f} sec")
    print(f"Max Latency        : {max_latency:.2f} sec")

if __name__ == "__main__":
    default_prompt = '''
# 根据 ##背景##、##问题## ，推导 ##思考过程##和##答案##：

##背景##
在布鲁克斯维尔这个小镇，当地政府最近决定实施一系列旨在促进经济增长和解决失业问题的经济政策。首项举措是大幅降低房主的房产税，旨在增加可支配收入并刺激消费者支出。同时，为吸引企业入驻，该镇向愿意在本地开展业务的公司提供税收优惠和补助。这一措施迅速促成了两家新制造厂在镇郊落户，承诺带来数百个就业岗位。然而，这些发展也引发了居民对于潜在环境影响以及对镇基础设施（包括道路和公共服务）压力的担忧。此外，住房价格出现上涨趋势，使一些长期居民即使享受了税收减免后，也难以继续负担住房费用。

##问题##
考虑到布鲁克斯维尔地方政府实施的经济政策，包括对房主的房产税减免和对企业的激励措施，这些政策整体上如何影响该镇的即时与长期社会经济状况，尤其是在就业、基础设施、环境影响和住房可负担性方面？

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=32, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent threads")
    parser.add_argument("--prompt", type=str, default=default_prompt, help="Prompt content")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens to generate")
    args = parser.parse_args()

    benchmark(args.requests, args.concurrency, args.prompt, args.max_tokens)

````

```yaml
#保持 cudagraph
vllm_enforce_eager: false
```

```
Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14989
Elapsed Time       : 58.29 sec
TPM (tokens/min)   : 15428.30
QPS (req/sec)      : 0.55
Avg Latency        : 13.49 sec
Max Latency        : 19.70 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14940
Elapsed Time       : 57.39 sec
TPM (tokens/min)   : 15620.31
QPS (req/sec)      : 0.56
Avg Latency        : 13.26 sec
Max Latency        : 19.70 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14684
Elapsed Time       : 56.11 sec
TPM (tokens/min)   : 15703.08
QPS (req/sec)      : 0.57
Avg Latency        : 12.86 sec
Max Latency        : 19.76 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14740
Elapsed Time       : 53.83 sec
TPM (tokens/min)   : 16430.55
QPS (req/sec)      : 0.59
Avg Latency        : 12.96 sec
Max Latency        : 18.64 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14860
Elapsed Time       : 58.01 sec
TPM (tokens/min)   : 15370.50
QPS (req/sec)      : 0.55
Avg Latency        : 13.12 sec
Max Latency        : 18.21 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14738
Elapsed Time       : 60.12 sec
TPM (tokens/min)   : 14707.63
QPS (req/sec)      : 0.53
Avg Latency        : 12.90 sec
Max Latency        : 17.96 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14798
Elapsed Time       : 61.54 sec
TPM (tokens/min)   : 14428.16
QPS (req/sec)      : 0.52
Avg Latency        : 12.96 sec
Max Latency        : 19.37 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 15155
Elapsed Time       : 58.56 sec
TPM (tokens/min)   : 15526.40
QPS (req/sec)      : 0.55
Avg Latency        : 13.61 sec
Max Latency        : 20.69 sec

```

```yaml
vllm_enforce_eager: false
infer_backend: vllm
vllm_extra_config:
  enable_chunked_prefill: true
  block_size: 32
  gpu_memory_utilization: 0.98
  max_num_batched_tokens: 24576

```
```bash
API_CONFIG_PATH=examples/inference/llama3_vllm.yaml llamafactory-cli api
```

```
Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14584
Elapsed Time       : 55.58 sec
TPM (tokens/min)   : 15743.40
QPS (req/sec)      : 0.58
Avg Latency        : 12.21 sec
Max Latency        : 18.21 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 15189
Elapsed Time       : 55.82 sec
TPM (tokens/min)   : 16325.61
QPS (req/sec)      : 0.57
Avg Latency        : 13.21 sec
Max Latency        : 17.87 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14735
Elapsed Time       : 55.83 sec
TPM (tokens/min)   : 15836.86
QPS (req/sec)      : 0.57
Avg Latency        : 12.79 sec
Max Latency        : 20.45 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14788
Elapsed Time       : 53.88 sec
TPM (tokens/min)   : 16466.55
QPS (req/sec)      : 0.59
Avg Latency        : 12.66 sec
Max Latency        : 23.99 sec

Successful Requests: 32
Total Tokens       : 14566
Elapsed Time       : 52.50 sec
TPM (tokens/min)   : 16646.24
QPS (req/sec)      : 0.61
Avg Latency        : 12.22 sec
Max Latency        : 16.35 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14756
Elapsed Time       : 55.12 sec
TPM (tokens/min)   : 16061.37
QPS (req/sec)      : 0.58
Avg Latency        : 12.56 sec
Max Latency        : 19.76 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14473
Elapsed Time       : 53.30 sec
TPM (tokens/min)   : 16292.29
QPS (req/sec)      : 0.60
Avg Latency        : 12.07 sec
Max Latency        : 16.93 sec

Total Requests     : 32
Successful Requests: 32
Total Tokens       : 14806
Elapsed Time       : 54.35 sec
TPM (tokens/min)   : 16345.09
QPS (req/sec)      : 0.59
Avg Latency        : 12.63 sec
Max Latency        : 22.23 sec

```


- 优化后的代码：
```python
# src/llamafactory/api/app.py
import yaml  # ✅ 新增导入

def run_api() -> None:
    # ✅ 读取默认 YAML 配置文件路径
    config_path = os.getenv("API_CONFIG_PATH", "examples/inference/llama3_vllm.yaml")
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)

    # ✅ 将 args 传入 ChatModel
    chat_model = ChatModel(args=args)

    app = create_app(chat_model)
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    print(f"Visit http://localhost:{api_port}/docs for API document.")
    uvicorn.run(app, host=api_host, port=api_port)

```

```python
#src/llamafactory/chat/chat_model.py
def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        args = args or {}

        # ✅ 关键改动：从 args 中弹出 vllm_extra_config，避免非法字段报错
        vllm_extra_config = args.pop("vllm_extra_config", {})
        
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        
        if model_args.infer_backend == EngineName.VLLM:
            self.engine: BaseEngine = VllmEngine(
                model_args=model_args,
                data_args=data_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                **vllm_extra_config
            )

        elif model_args.infer_backend == EngineName.VLLM:
            self.engine: BaseEngine = VllmEngine(
                model_args=model_args,
                data_args=data_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                **vllm_extra_config  # ✅ 传入额外参数
            )
        elif model_args.infer_backend == EngineName.SGLANG:
            self.engine: BaseEngine = SGLangEngine(model_args, data_args, finetuning_args, generating_args)
        else:
            raise NotImplementedError(f"Unknown backend: {model_args.infer_backend}")
    
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()
```

```python
#src/llamafactory/chat/vllm_engine.py
def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        block_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        max_num_batched_tokens: Optional[int] = None,
        enable_chunked_prefill=None,
        tensor_parallel_size=None,         
        pipeline_parallel_size=None,  
    ) -> None:
        self.name = EngineName.VLLM
        self.model_args = model_args
        config = load_config(model_args)  # may download model from ms hub
        if getattr(config, "quantization_config", None):  # gptq models should use float16
            quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
            quant_method = quantization_config.get("quant_method", "")
            if quant_method == QuantizationMethod.GPTQ and model_args.infer_dtype == "auto":
                model_args.infer_dtype = "float16"

        self.can_generate = finetuning_args.stage == "sft"
        tokenizer_module = load_tokenizer(model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        self.tokenizer.padding_side = "left"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        self.template.mm_plugin.expand_mm_tokens = False  # for vllm generate
        self.generating_args = generating_args.to_dict()

        engine_args = {
            "model": model_args.model_name_or_path,
            "trust_remote_code": model_args.trust_remote_code,
            "download_dir": model_args.cache_dir,
            "dtype": model_args.infer_dtype,
            "max_model_len": model_args.vllm_maxlen,
            "tensor_parallel_size": get_device_count() or 1,
            "gpu_memory_utilization": model_args.vllm_gpu_util,
            "disable_log_stats": True,
            "disable_log_requests": True,
            "enforce_eager": model_args.vllm_enforce_eager,
            "enable_lora": model_args.adapter_name_or_path is not None,
            "max_lora_rank": model_args.vllm_max_lora_rank,
            
        }

        # ✅ 注入用户配置的额外推理参数
        if block_size is not None:
            engine_args["block_size"] = block_size
        if gpu_memory_utilization is not None:
            engine_args["gpu_memory_utilization"] = gpu_memory_utilization
        if max_num_batched_tokens is not None:
            engine_args["max_num_batched_tokens"] = max_num_batched_tokens
        if enable_chunked_prefill is not None:
            engine_args["enable_chunked_prefill"] = enable_chunked_prefill    
        if tensor_parallel_size is not None:
            engine_args["tensor_parallel_size"] = tensor_parallel_size
        if pipeline_parallel_size is not None:
            engine_args["pipeline_parallel_size"] = pipeline_parallel_size

        if self.template.mm_plugin.__class__.__name__ != "BasePlugin":
            engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2}

        if isinstance(model_args.vllm_config, dict):
            engine_args.update(model_args.vllm_config)

        if getattr(config, "is_yi_vl_derived_model", None):
            import vllm.model_executor.models.llava

            logger.info_rank0("Detected Yi-VL model, applying projector patch.")
            vllm.model_executor.models.llava.LlavaMultiModalProjector = LlavaMultiModalProjectorForYiVLForVLLM

        self.model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
        if model_args.adapter_name_or_path is not None:
            self.lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
        else:
            self.lora_request = None
```