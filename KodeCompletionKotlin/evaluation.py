import jsonlines
import json
import matplotlib.pyplot as plt
from mxeval.evaluation import evaluate_functional_correctness
import pandas as pd


def kotlin_human_eval(generated_data:str):
    #task_id,prompt,entry_point,test
    df = pd.read_parquet("hf://datasets/JetBrains/Kotlin_HumanEval/data/train-00000-of-00001.parquet")
    problem_dict = {problem['task_id']: problem for _,problem in df.iterrows()}


    eval_base = []
    with jsonlines.open(generated_data, mode="r") as reader:
        for line in reader:
            eval_base.append(line)


    evaluate_functional_correctness(
        sample_file=generated_data,
        k=[1],
        n_workers=16,
        timeout=50,
        problem_file=problem_dict,

    )

    with open(generated_data + '_results.jsonl') as fp:
        total = 0
        correct = 0
        for line in fp:
            sample_res = json.loads(line)
            total += 1
            correct += sample_res['passed']

    return correct/total

def vizualize(first_model_data:dict,second_model_data:dict):
    plt.barh(first_model_data['name'],first_model_data["score"],color="green")
    plt.barh(second_model_data['name'],second_model_data["score"],color="red")
    plt.title("Models score based on Human Eval")
    plt.xlabel("Score")
    plt.ylabel("model")
    plt.tight_layout()
    plt.show()
