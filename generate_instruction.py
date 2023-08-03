"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils

import fire

# import pprint

def encode_prompt(prompt, prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, output) = task_dict["instruction"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["message"]["content"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            print('Skipping the instruction because it is likely truncated.')
            print(f'Instruction:', inst)
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Output):", inst)
        ## the splitted_data should have length of 7: ['', 'Instruction', {instruction}, 'Input', {input}, 'Output', {output}]
        # the splitted_data should have length of 5: ['', 'Instruction', {instruction}, 'Output', {output}]
        if len(splitted_data) != 5:
            print('Skipping the instruction because splitted_data length is not 5.')
            print(f'Instruction:', inst)
            continue
        else:
            inst = splitted_data[2].strip()
            # input = splitted_data[4].strip()
            # input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[4].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            print('Skipping the instruction because it is too short or too long.')
            print(f'Instruction:', inst)
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            # "image",
            # "images",
            "graph",
            "graphs",
            # "picture",
            # "pictures",
            # "file",
            # "files",
            # "map",
            # "maps",
            "draw",
            "plot",
            "go to",
            # "video",
            "audio",
            # "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            print('Skipping the instruction because it contains blacklisted words.')
            print(f'Instruction:', inst)
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            print('Skipping the instruction because it starts with "Write a program".')
            print(f'Instruction:', inst)
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            print('Skipping the instruction because it starts with punctuation.')
            print(f'Instruction:', inst)
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            print('Skipping the instruction because it starts with non-english character.')
            print(f'Instruction:', inst)
            continue
        instructions.append({"instruction": inst, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    num_instructions_to_generate,
    num_instructions_to_generate_per_request,
    output_dir="./",
    seeds_dir="./seeds",
    instruction_path="./instruction.txt",
    model_name="gpt-3.5-turbo",
    num_prompt_instructions=3,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    similarity_threshold=0.7,
):
    with open(instruction_path, 'r') as file:
      instruction = file.read()
    
    seed_instruction_data = []
    seed_paths = os.listdir(seeds_dir)
    for seed_path in seed_paths:
      # Check if the path is a file (not a directory)
      if not os.path.isfile(os.path.join(seeds_dir, seed_path)): continue
      
      seed_website_type, file_ext = os.path.splitext(seed_path)
      
      # Check if the file is HTML file
      if file_ext.lower() != '.html': continue
      
      with open(os.path.join(seeds_dir, seed_path), 'r') as f:
        seed_output = f.read()
        seed = {
          # Format the instruction template with the dynamic substring
          'instruction': instruction.format(website_type=seed_website_type.replace('-', ' ')), 
          'output': seed_output,
        }
        seed_instruction_data.append(seed)
    
    print(f"Loaded {len(seed_instruction_data)} seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        num_tasks = num_prompt_instructions + num_instructions_to_generate_per_request
        prompt_to_encode = open("./prompt.txt").read() + "\n"
        prompt_to_encode = prompt_to_encode.replace('{num_tasks}', str(num_tasks))
        prompt_to_encode = prompt_to_encode.replace('{first_generated_instruction_num}', str(num_prompt_instructions+1))
        prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
        prompt = encode_prompt(prompt_to_encode, prompt_instructions)
        
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=6400,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            # stop=[f"\n{num_tasks}", f"{num_tasks}.", f"{num_tasks}."],
        )
        request_start = time.time()
        result = utils.openai_completion(
            prompt=prompt,
            model_name=model_name,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        
        new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
        instruction_data += new_instructions
        
        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            # Check the similarity
            if max(rouge_scores) > similarity_threshold:
                print(f'Skipping the instruction because the similarity score: {max(rouge_scores)} > {similarity_threshold}.')
                print(f'Instruction:', instruction_data_entry["instruction"])
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s, {result['total_tokens']} tokens")
        print(f"Generated {total} instructions, kept {keep} instructions")
        
        # If total generated is 0, print the result, because maybe something's wrong
        if total == 0:
            print(f'Result:', result)
        
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)