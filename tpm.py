import time
import requests
import concurrent.futures
import argparse
import os
from datetime import datetime

url = "http://localhost:8000/v1/chat/completions"
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
        response = requests.post(url, json=data, headers=headers, timeout=60)
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