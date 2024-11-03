import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer,StoppingCriteria,StoppingCriteriaList
from datetime import datetime
import pandas as pd
import jsonlines
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer):
        (StoppingCriteria.__init__(self),)
        self.stops = rf"{stops}"
        self.tokenizer = tokenizer
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_three_tokens = [int(x) for x in input_ids.data[0][-3:]]
        decoded_last_three_tokens = self.tokenizer.decode(last_three_tokens)

        return bool(re.search(self.stops, decoded_last_three_tokens))

#function to clear any comments from answer
def clean_answer(code):
    # Clean comments
    code_without_line_comments = re.sub(r"//.*", "", code)
    code_without_all_comments = re.sub(
        r"/\*.*?\*/", "", code_without_line_comments, flags=re.DOTALL
    )
    # Clean signatures
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("fun "):
            return "\n".join(lines[i + 1:])

    return code

#importing data for fine-tuning and testing

#task,id,prompt,entry_point,test
def generate_data(test_data,language:str,model_path:str,device:str):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    #switches to evalmode freez batch norm,turn off dropout
    model.eval()

    results = []
    for i in range(len(test_data)):
        print(f"Generating {i+1} data out of {len(test_data)}")
        criterion = StoppingCriteriaSub(stops="\n}\n", tokenizer=tokenizer)
        stopping_criteria = StoppingCriteriaList([criterion])
        task = tokenizer.encode(test_data["prompt"][i], return_tensors="pt",padding=True).to(device)
        sample = model.generate(
            task,
            max_new_tokens=256,
            min_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
            stopping_criteria=stopping_criteria
        )
        answer = clean_answer(tokenizer.decode(sample[0], skip_special_tokens=True))
        results.append({"task_id":test_data["task_id"][i],"completion": answer,"language":language})

    now = datetime.now()
    output_file = f"answers_from_"+str(model_path.split("/")[-1])+"_date_"+now.strftime("%d_%m_%Y %H_%M_%S")
    with jsonlines.open(output_file, mode="w") as writer:
        for line in results:
            writer.write(line)


#jak benchmarka zrobic: dodac slownik id,prompt,odpowiedz #policzyc z tego bleu score
