import jsonlines
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import gc
#Additional Info when using cuda
torch.cuda.empty_cache()
gc.collect()
import os



class PromptAnswerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        prompt = self.data[idx]["prompt"]
        answer = self.data[idx]["solution"]

        # Prepare inputs and labels by concatenating prompt and answer
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length
        )
        labels = self.tokenizer(
            answer, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length
        )

        # Align input and labels to match causal LM expectations
        inputs["input_ids"] = inputs["input_ids"].squeeze()
        labels["input_ids"] = labels["input_ids"].squeeze()

        return {
            "input_ids": inputs["input_ids"],
           # "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }
def fine_tune(fine_tuning_data_translated:str,fine_tuning_data:str,model_path:str,model_name:str,device:str="cuda",no_cuda:bool=True):

    fine_tuning_set = []
    with jsonlines.open(fine_tuning_data_translated, mode="r") as reader:
        for i,line in enumerate(reader):
            fine_tuning_set.append({"prompt":fine_tuning_data[i],"solution":line})

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    fine_tuning_set = PromptAnswerDataset(fine_tuning_set,tokenizer)
    train_arguments = TrainingArguments(
        output_dir=str(os.path.abspath(os.getcwd()))+"\\FineTunes\\"+model_name,
        learning_rate=2e-5,
        num_train_epochs=4,
        weight_decay=0.2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        no_cuda=no_cuda

    )
    trainer = Trainer(
        model=model,
        args=train_arguments,
        train_dataset=fine_tuning_set,
        tokenizer = tokenizer

    )

    torch.cuda.memory_summary(device=None, abbreviated=False)
    trainer.train()

