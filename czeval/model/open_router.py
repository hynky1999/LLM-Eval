import aiohttp
import asyncio
from functools import partial
import json
import logging
import time
from tqdm.asyncio import tqdm_asyncio


from czeval.common import OPENROUTER_API_KEY
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)


@retry(
    stop=stop_after_attempt(30),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=(
        retry_if_exception_type(
            aiohttp.ClientError | TimeoutError | asyncio.TimeoutError | RetryError
        )
    ),
)
async def predict_sample(session, conversation, model, temp, max_tokens):
    async with session.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": model,
                "messages": conversation,
                "temperature": temp,
                "max_tokens": max_tokens,
            }
        ),
    ) as response:
        if response.status != 200:
            raise Exception(f"Invalid response: {response.text}")
        data = await response.json()
        if "error" in data:
            logging.error(data["error"])
            if data["error"].get("code") in [429, 502, 503]:
                raise ValueError(f"Rate limited: {data['error']}")
            raise Exception(f"Invalid response: {data['error']}")
        return data["choices"][0]["message"]["content"]


def rate_limiter(max_calls_per_second):
    min_interval = 1.0 / float(max_calls_per_second)

    def decorate(func):
        last_time_called = [0.0]

        async def rate_limited_function(*args, **kwargs):
            while (
                min_interval
                - (time_to_wait := time.perf_counter() - last_time_called[0])
                > 0
            ):
                await asyncio.sleep(time_to_wait)

            last_time_called[0] = time.perf_counter()
            ret = await func(*args, **kwargs)
            return ret

        return rate_limited_function

    return decorate


async def handle_failure(func, *args, **kwargs):
    try:
        return await func(*args, **kwargs)
    except RetryError as e:
        logging.error(e)
        return ""


def predict_samples(
    conversations: list[dict],
    model: str,
    temp: float,
    max_tokens: int,
    max_requests_per_second: int = 1,
):
    rate_limited_predict = rate_limiter(max_requests_per_second)(
        partial(
            handle_failure,
            predict_sample,
            model=model,
            temp=temp,
            max_tokens=max_tokens,
        )
    )

    async def get_predictions():
        async with aiohttp.ClientSession() as session:
            return await tqdm_asyncio.gather(
                *[rate_limited_predict(session, conv) for conv in conversations]
            )

    predictions = asyncio.run(get_predictions())
    return predictions
