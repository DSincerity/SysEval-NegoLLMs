import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import arguments
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datetime import datetime, timezone, timedelta
from functools import partial
from sklearn.metrics import accuracy_score
from baselines.tasks.create_task_dataset import CreateDatasetByTasks
from baselines.tasks.task_selector import TaskSelector
from baselines.tasks.prompt_modifier import PromptModifier
from baselines.datasets.dataset import RawDatesetLoader
from utils import write_json


def preprocess_function(sample,  max_source_length, max_target_length, padding="max_length"):
    # tokenize inputs
    model_inputs = tokenizer(sample["input"], max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(sample["label"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = {"acc": accuracy_score(decoded_preds, decoded_labels)}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

if __name__ == '__main__':

    parser = ArgumentParser()
    parser = arguments.add_arguments(parser)
    parser = arguments.add_model_arguments(parser)
    parser = arguments.add_model_hyperparameters(parser)
    args = parser.parse_args()

    print("="*10, "input arguments", "="*10)
    print(args)

    local_rank = 0
    # GPU
    if args.do_train:
        local_rank = os.environ.get("LOCAL_RANK", 0)
        print(f"local_rank:{local_rank}")
        device = torch.device(
            "cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if local_rank == 0:
        print(args)
        print(f">>> Use Device: {device}")

    # Train Serial
    pst = timezone(timedelta(hours=-8))
    time_info = datetime.now(tz=pst).strftime("%Y%m%d-%H:%M")

    #######################
    # Load dataset
    #######################
    # Load tokenizer
    if args.checkpoint is not None:
        print("Load tokenizer from checkpoint ")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.add_tokens(['{', '}'], special_tokens=False)

    whole_datasset=dict()
    task_selector = TaskSelector()
    if args.task_name:
        task_nm_list = [args.task_name]
    else:
        task_nm_list = task_selector.get_tasks_by_dataset_type(args.task_dataset, args.task_type)

    data_full_nm_mapper= {'cra': 'cra', 'dnd': 'dnd', 'ji': 'job_interview', 'ca': 'casino'}
    load_datasets= {dataset: [task for task in task_nm_list if f"_{dataset}" in task] \
                    for dataset in set([task.replace("ji_w","ji").split("_")[-1] for task in task_nm_list])}
    print("\nTasks per dataset: ", load_datasets)

    org_dataset = RawDatesetLoader(list(load_datasets.keys()), storage_dir=args.storage_dir).merged_dataset
    prompt_modifier = PromptModifier()
    for dataset_nm, tasks in load_datasets.items():
        print("Dataset: ", dataset_nm, ", Tasks: ", tasks)

        for _task in tqdm(tasks):
            print("*"*10,_task,"*"*10)
            args.task_name, args.dataset_name = _task, data_full_nm_mapper[_task.replace("ji_w","ji").split("_")[-1]]
            df = CreateDatasetByTasks(args, args.task_name, args.model_name, args.dataset_name, org_dataset)

            # Prompt modification
            print('Prompt modification start...')
            df.train_input_text = list(map(partial(prompt_modifier.modify, task_nm=_task), df.train_input_text))
            df.test_input_text = list(map(partial(prompt_modifier.modify, task_nm=_task), df.test_input_text))
            print('Prompt modification Done')

            # get max token length
            max_src_len, max_trg_len = df.get_max_len(tokenizer)

            # Save dataset
            whole_datasset[_task] = dict()
            whole_datasset[_task]['train'] = (df.train_input_text, df.train_labels)
            whole_datasset[_task]['test'] = (df.test_input_text, df.test_labels)
            whole_datasset[_task]['max_len'] = (max_src_len, max_trg_len)
            print(f'[{_task}] Original train size: {len(df.train_input_text)}, test size : {len(df.test_input_text)} \
                max token: {max_src_len}, {max_trg_len}')
            print(f"[{_task}] Adjusted train size: {len(whole_datasset[_task]['train'][0])}, test size : {len(whole_datasset[_task]['test'][0])} \
                max token: {max_src_len}, {max_trg_len}")
            print("="*10, "train example", "="*10)
            print(df.train_input_text[1])
            print("="*10, "train label example", "="*10)
            print(df.train_labels[1])
            print()
            del df

    merge_train = {'input_text': [], 'label': []}
    merge_test = {'input_text': [], 'label': []}
    for task_nm, data in whole_datasset.items():

        # modify prompt & merge datasets
        merge_train['input_text'].extend(data['train'][0])
        merge_train['label'].extend(data['train'][1])
        merge_test['input_text'].extend(data['test'][0])
        merge_test['label'].extend(data['test'][1])

    train= Dataset.from_list([{"input": _text, 'label': _label} for _text, _label in zip(merge_train['input_text'], merge_train['label'])])
    test = Dataset.from_list([{"input": _text, 'label': _label} for _text, _label in zip(merge_test['input_text'], merge_test['label'])])

    max_src_toks= max([v['max_len'][0] for k , v in whole_datasset.items()])
    max_trg_toks=max([v['max_len'][1] for k , v in whole_datasset.items()])

    preprocess= partial(preprocess_function, max_source_length=max_src_toks, max_target_length=max_trg_toks)

    tokenized_train_dataset = train.map(preprocess, batched=True, remove_columns=["input", "label"]).shuffle()
    tokenized_test_dataset = test.map(preprocess, batched=True, remove_columns=["input", "label"]).shuffle()

    #######################
    # Model training
    #######################
    # Model
    if args.checkpoint is not None:
        print("Load model from checkpoint ")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        model.resize_token_embeddings(len(tokenizer))

    # ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        predict_with_generate=True,
        gradient_accumulation_steps=args.grad_accum_step,
        fp16=False, # Overflows with fp16
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=args.loging_step,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        save_steps=args.eval_step,
        eval_steps=args.eval_step,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        overwrite_output_dir=False,
        push_to_hub=False,
    )

    #early_stop = EarlyStoppingCallback(3, 1.0) # patience, threshold
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        #callbacks=[early_stop]
    )

    # Training
    if args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        #trainer.save_model()  # Saves the tokenizer too for easy upload

        print("Start Training")
        metrics = train_result.metrics
        metrics["train_samples"] = len(trainer.train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        print("Start Evaluation")
        if args.overall_eval:
            metrics = trainer.evaluate()
            metrics[f"eval_samples"] = len(trainer.eval_dataset)
            metrics['tasks'] = whole_datasset.keys()
            trainer.log_metrics(f"eval", metrics)
            trainer.save_metrics(f"eval", metrics)
        else:
            # evaluate per tasks
            eval_test = dict()
            final_results = dict()
            for task_nm, data in whole_datasset.items():
                    print("task: ", task_nm)
                    eval_test['input_text']=list(map(partial(prompt_modifier.modify, task_nm=task_nm), data['test'][0]))
                    eval_test['label']=data['test'][1]

                    evalset = Dataset.from_list([{"input": _text, 'label': _label} for _text, _label in zip(eval_test['input_text'], eval_test['label'])]).shuffle()
                    max_src_toks, max_trg_toks= data['max_len']

                    preprocess= partial(preprocess_function, max_source_length=max_src_toks, max_target_length=max_trg_toks)
                    tokenized_eval_dataset = evalset.map(preprocess, batched=True, remove_columns=["input", "label"])
                    trainer.eval_dataset = tokenized_eval_dataset

                    metrics = trainer.evaluate()
                    metrics[f"eval_{task_nm}_samples"] = len(trainer.eval_dataset)
                    final_results[task_nm] = metrics
                    #trainer.log_metrics(f"eval_{task_nm}", metrics)
                    #trainer.save_metrics(f"eval_{task_nm}", metrics)
            write_json(final_results, os.path.join(args.output_dir,f'eval_results_per_task.{os.path.basename(args.output_dir)}.json'))
            print("Evaluation Done: saved to ", os.path.join(args.output_dir,f'eval_results_per_task.{os.path.basename(args.output_dir)}.json'))

    if args.do_generate:
        print("Start Generation")
        if args.overall_eval:
            predict_results = trainer.predict(trainer.eval_dataset, metric_key_prefix="predict")
            decoded_output = tokenizer.batch_decode(predict_results[0], skip_special_tokens=True)
            results=dict()
            results['tasks'] = whole_datasset.keys()
            results[f"gen_{task_nm}_samples"] = len(trainer.eval_dataset)
            results["generations"] = str(decoded_output)
            trainer.log_metrics(f"gen_{task_nm}", results)
            trainer.save_metrics(f"gen_{task_nm}", results)
        else:
            # generate per tasks
            eval_test = dict()
            final_results = dict()
            for task_nm, data in whole_datasset.items():
                    print("task: ", task_nm)
                    eval_test['input_text']=list(map(partial(prompt_modifier.modify, task_nm=task_nm), data['test'][0]))
                    eval_test['label']=data['test'][1]

                    evalset = Dataset.from_list([{"input": _text, 'label': _label} for _text, _label in zip(eval_test['input_text'], eval_test['label'])]).shuffle()
                    max_src_toks, max_trg_toks= data['max_len']

                    preprocess= partial(preprocess_function, max_source_length=max_src_toks, max_target_length=max_trg_toks)
                    tokenized_eval_dataset = evalset.map(preprocess, batched=True, remove_columns=["input", "label"])
                    trainer.eval_dataset = tokenized_eval_dataset

                    predict_results = trainer.predict(trainer.eval_dataset, metric_key_prefix="predict")
                    decoded_output = tokenizer.batch_decode(predict_results[0], skip_special_tokens=True)
                    results=dict()
                    results[f"gen_{task_nm}_samples"] = len(trainer.eval_dataset)
                    results["generations"] = str(decoded_output)
                    final_results[task_nm] = results
                    #trainer.log_metrics(f"gen_{task_nm}", results)
                    #trainer.save_metrics(f"gen_{task_nm}", results)
            write_json(final_results, os.path.join(args.output_dir, f'gen_results_per_task.{os.path.basename(args.output_dir)}.json'))
            print("Generation Done: saved to ", os.path.join(args.output_dir, f'gen_results_per_task.{os.path.basename(args.output_dir)}.json'))
