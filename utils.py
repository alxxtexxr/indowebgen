import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

import openai
import tqdm
from openai import openai_object
import copy

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

# openai_org = os.getenv("OPENAI_ORG")
# if openai_org is not None:
#     openai.organization = openai_org
#     logging.warning(f"Switching to organization: {openai_org} for OAI API key.")

# OpenAI API authentication
with open('secrets.json') as f:
    secrets = json.load(f)

if 'OPENAI_API_KEY' not in secrets:
  print('OpenAI API keys not found.')
  
openai.api_key = secrets['OPENAI_API_KEY']

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def openai_completion(
    prompt: str,
    decoding_args: OpenAIDecodingArguments,
    model_name="gpt-3.5-turbo",
    sleep_time=2,
    **decoding_kwargs,
) -> StrOrOpenAIObject:
    """Decode with OpenAI API.

    Args:
        prompts: A string to complete. 
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion, the completion type is an openai_object.OpenAIObject object.
    """
    completion = {}
    batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

    while True:
        try:
            shared_kwargs = dict(
                model=model_name,
                **batch_decoding_args.__dict__,
                **decoding_kwargs,
            )
            # If using chat model, use ChatCompletion
            completion_batch = openai.ChatCompletion.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                **shared_kwargs,
            )
            choice = completion_batch.choices[0]
            choice["total_tokens"] = completion_batch.usage.total_tokens
            completion = choice
            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.
    return completion


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict