# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import asyncio

import random
import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker

from .protocol import Request


@triton_worker()
async def worker(
    runtime: DistributedRuntime, prompts: list[str], max_tokens: int, temperature: float
):
    """
    Instantiate a `backend` client and call the `generate` endpoint with load balancing
    """
    # get endpoint
    endpoint = runtime.namespace("triton-init").component("vllm").endpoint("generate")

    # create client
    client = await endpoint.client()
    
    # Track in-progress requests per endpoint
    endpoint_loads = {endpoint_id: 0 for endpoint_id in client.endpoint_ids()}

    async def process_request(prompt: str):
        await asyncio.sleep(random.random())
        
        # Find endpoint with minimum load
        endpoint_id = min(endpoint_loads.items(), key=lambda x: x[1])[0]
        endpoint_loads[endpoint_id] += 1
        
        print(endpoint_loads)
        
        try:
            request = Request(
                prompt=prompt,
                sampling_params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            ).model_dump_json()
            
            async for resp in await client.direct(request, endpoint_id):
                pass
        finally:
            endpoint_loads[endpoint_id] -= 1

    # Create tasks for all prompts
    tasks = [process_request(prompt) for prompt in prompts]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    uvloop.install()

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, action='append', 
                       help="Can be specified multiple times for multiple prompts")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.5)

    args = parser.parse_args()

    # Use default prompt if none provided
    prompts = ["Write me a short story about a capybara playing the violin."]
    prompts = prompts * 100
    
    asyncio.run(worker(prompts, args.max_tokens, args.temperature))