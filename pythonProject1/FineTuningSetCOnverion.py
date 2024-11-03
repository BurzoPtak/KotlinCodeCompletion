import time

import pandas as pd
import os
from openai import OpenAI,RateLimitError
from datasets import load_dataset
import jsonlines


def data_conversion(api_key:str,language:str,data_to_convert:str,model:str="gpt-3.5-turbo",
                    output:str=f"fine_tuning_data",retries:int=3,delay:int = 8):
    os.environ['OPENAI_API_KEY'] = api_key
    client = OpenAI()
    system_prompt = "ignoring Markdown translate this script into "+language+" language, skip comments and do not write any:"

    results = []
    for i in data_to_convert:
        prompt = i
        main_user_prompt = prompt.strip()
        for attempt in range(retries):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                {"role": "user", "content": main_user_prompt}
                    ],
                    temperature=1,
                    max_tokens=1024,

                )
                results.append(completion.choices[0].message.content)
                break
            except RateLimitError:
                if attempt <retries:
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(str(RateLimitError))

    output_file = output
    with jsonlines.open(output_file, mode="w") as writer:
        for line in results:
            writer.write(line)