import pandas as pd
from GenerateData import generate_data
from FineTuning import fine_tune
from FineTuningSetCOnverion import data_conversion
from datasets import load_dataset
from evaluation import vizualize,kotlin_human_eval

#Data to translate from kotlin to python
api_key ="YOUR API KEY"
ds = load_dataset("jinaai/code_exercises")
ds_conversion = ds["train"].to_pandas()
ds_conversion = ds_conversion['solution'].to_list()[:100]

#Data to generate data from model from prompt
prompt_data = pd.read_parquet("hf://datasets/JetBrains/Kotlin_HumanEval/data/train-00000-of-00001.parquet")
language = "kotlin"
device = "cpu" #or cuda for gpu
model_path = "ibm-granite/granite-3b-code-base-2k"
fine_tuned_model_path = "FineTunes/Your model"

#Data for fine tuning
fine_tuning_data = "fine_tuning_data"
ds_fine_tune = ds["train"].to_pandas()
ds_fine_tune = ds_fine_tune["problem"].to_list()
ds_fine_tune = ds_fine_tune[:100]
model_name = "fineTuned"
#Data for human evaluation
generated_data = "Generated/answers"
generated_fine_tuned_data = "answers_from_checkpoint-52_date_03_11_2024 16_12_45"



while True:
    choice = input("""
    What would you like to do?
    1)convert data from python to kotlin using chatGPT
    2)Generate answers to coding tasks in kotlin using one of 2 models
    3)Fine tune models
    4)test it on Human evaluation set and vizualize results between standard model and fine tuned one
    """)

    if choice == "1":
        ch2 = input("would you like to change parameters from default?")
        if ch2 == "y":
            api_key = input("please specify your api key")
            language = input("please specify language")
            k = input("ds_conversion was implemented directly in the code press anything to continue")
            data_conversion(api_key, language, ds_conversion)
        else:
            data_conversion(api_key,language,ds_conversion)

    if choice == "2":
        ch2 = input("would you like to change parameters from default?")
        if ch2 == "y":
            prompt_data = input("Please enter your data containing prompts should be [prompt] dict")
            language = input("please specify language to which it should be translated")
            model_path = input("Please specify path to your model")
            device = input("cuda for GPU or cpu ")
            generate_data(prompt_data,language,fine_tuned_model_path,device)
        else:
            generate_data(prompt_data,language,model_path,device)

    if choice == "3":
        ch2 = input("would you like to change parameters from default?")
        if ch2 == "y":
            fine_tuning_data = input("specify path for generated answers")
            k = input("ds_fine_tune was implemented directly in the code press anything to continue")
            model_path = input("specify path to core model")
            model_name = input("name your model :)")
            fine_tune(fine_tuning_data,ds_fine_tune,model_path,model_name)
        else:
            fine_tune(fine_tuning_data,ds_fine_tune,model_path,model_name)

    if choice == "4":
        ch2 = input("would you like to change parameters from default?")
        if ch2 == "y":
            model_path = input("specify path to first model")
            generated_data = input("specify path to answers from said model")
            fine_tuned_model_path = input("specify path to first model")
            generated_fine_tuned_data = input("specify path to answers from said model")
            first_model_data = {'name':str(model_path.split("/")[-1]),"score":kotlin_human_eval(generated_data)}
            second_model_data = {'name':str(fine_tuned_model_path.split("/")[-1]),"score":kotlin_human_eval(generated_fine_tuned_data)}
            vizualize(first_model_data,second_model_data)
        else:
            first_model_data = {'name': str(model_path.split("/")[-1]), "score": kotlin_human_eval(generated_data)}
            second_model_data = {'name': str(fine_tuned_model_path.split("/")[-1]), "score": kotlin_human_eval(generated_fine_tuned_data)}
            vizualize(first_model_data, second_model_data)
    else:
        break


