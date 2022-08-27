import os
import json
import random
import torch
from datetime import datetime
from evaluation.tasks import MultiChoiceTask, GenerationTask
from evaluation.tasks import MultiChoiceTaskDataset, GenerationTaskDataset, MultiChoiceTaskConfig, GenerationTaskConfig
from SwissArmyTransformer import get_tokenizer
from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from typing import Dict, Union, List
from evaluation import print_rank_0


def build_vqa_prompt(item, entities=None, ocr_results=None, captions=None):
    image_id = item["question_id"]
    overall_sent = f'Please observe the image and answer the question.'
    caption_sent = None
    if captions is not None:
        caption = captions[image_id]["caption"]
        caption_sent = f"This image is {caption}"
    scene_sent, object_sent = None, None
    if entities is not None:
        description = entities[image_id]
        img_type, ppl_result, sorted_places = description["img_type"], description["ppl_result"], description[
            "sorted_places"]
        object_list = description["object_list"]
        object_result = ", ".join(object_list)
        scene_sent = f'I think this image was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.'
        object_sent = f'I think there might be a {object_result} in this {img_type}.'
    ocr_sent = None
    if ocr_results is not None:
        ocr_result = ocr_results[image_id]
        if ocr_result["texts"]:
            texts = sorted(ocr_result["texts"], key=lambda x:x["confidence"], reverse=True)
            texts = ", ".join([f'"{item["text"]}"' for item in texts[:5]])
            ocr_sent = f'The texts in the image include {texts}.'
    prompts = [overall_sent]
    if caption_sent is not None:
        prompts.append(caption_sent)
    if scene_sent is not None:
        prompts.append(scene_sent)
    if ocr_sent is not None:
        prompts.append(ocr_sent)
    if object_sent is not None:
        prompts.append(object_sent)
    prompt = " ".join(prompts)
    question = item["question"]
    prompt = prompt + question + " Answer:"
    return prompt


def load_descriptions(config, split):
    descriptions = {"clip": None, "ocr": None, "captions": None}
    if config.clip_pattern is not None:
        clip_file_path = config.clip_pattern[split]
        descriptions["clip"] = load_description_file(clip_file_path)
        print_rank_0(f"Loading {split} clip descriptions from {clip_file_path}")
    if config.ocr_pattern is not None:
        ocr_file_path = config.ocr_pattern[split]
        descriptions["ocr"] = load_description_file(ocr_file_path)
        print_rank_0(f"Loading {split} OCR results from {ocr_file_path}")
    if config.caption_pattern is not None:
        caption_file_path = config.caption_pattern[split]
        descriptions["captions"] = load_description_file(caption_file_path)
        print_rank_0(f"Loading {split} captions from {caption_file_path}")
    return descriptions


def build_priming_vqa_prompt(config, add_answer_period=False):
    priming_prompt = ""
    assert config.train_path is not None
    with open(config.train_path) as file:
        dataset = json.load(file)
    descriptions = load_descriptions(config, "train")
    for item in random.sample(dataset, config.num_train_examples):
        prompt = build_vqa_prompt(item, entities=descriptions["clip"], ocr_results=descriptions["ocr"],
                                  captions=descriptions["captions"])
        answer = item["choices"][item["correct_choice_idx"]]
        priming_prompt += prompt + answer
        if add_answer_period:
            priming_prompt += "."
        priming_prompt += " "
    tokenizer = get_tokenizer()
    prompt_length = len(tokenizer.tokenize(priming_prompt))
    print_rank_0(f"Priming prompt length {prompt_length}, content: {priming_prompt}")
    return priming_prompt


def load_description_file(path):
    descriptions = {}
    with open(path) as file:
        for line in file:
            description = json.loads(line)
            descriptions[description['image_id']] = description
    return descriptions


@dataclass
class VQAConfig(YAMLWizard):
    clip_pattern: Union[str, Dict[str, str]] = None  # Organize data file in groups
    ocr_pattern: Union[str, Dict[str, str]] = None
    caption_pattern: Union[str, Dict[str, str]] = None
    priming: bool = False
    num_train_examples: int = 10
    train_path: str = None


@dataclass
class VQAMulConfig(VQAConfig, MultiChoiceTaskConfig):
    pass


@dataclass
class VQAGenConfig(VQAConfig, GenerationTaskConfig):
    rationale_generation: bool = False
    pass


class VQAMulDataset(MultiChoiceTaskDataset):
    config: VQAMulConfig

    def __init__(self, path, split, config: VQAMulConfig):
        self.descriptions = load_descriptions(config, split)
        self.priming = config.priming
        self.num_train_examples = config.num_train_examples
        if self.priming:
            self.priming_prompt = build_priming_vqa_prompt(config)
        super().__init__(path, config)

    def process_single_item(self, item, **kwargs):
        image_id = item["question_id"]
        prompt = build_vqa_prompt(item, entities=self.descriptions["clip"], ocr_results=self.descriptions["ocr"],
                                  captions=self.descriptions["captions"])
        if self.priming:
            prompt = self.priming_prompt + prompt
        choices = item["choices"]
        if 'correct_choice_idx' in item:
            label = item["correct_choice_idx"]
        else:
            label = 0
        return super().process_single_item(
            {"inputs_pretokenized": prompt, "choices_pretokenized": choices, "label": label}, image_id=image_id,
            choices_pretokenized=choices,
            **kwargs)


class VQAMulTask(MultiChoiceTask):
    config: VQAMulConfig

    @classmethod
    def config_class(cls):
        return VQAMulConfig

    def build_dataset(self, relative_path, split):
        return VQAMulDataset(os.path.join(self.config.path, relative_path), split, self.config)

    def save_prediction_to_file(self, file, predictions, data):
        results = {}
        for prediction, item in zip(predictions, data):
            results[item["image_id"]] = {"multiple_choice": item["choices_pretokenized"][prediction]}
        with open("outputs/prediction_mul_" + datetime.now().strftime('%m-%d-%H-%M_') + file, "w") as output:
            json.dump(results, output)


class VQAGenDataset(GenerationTaskDataset):
    config: VQAGenConfig

    def __init__(self, path, split, config: VQAGenConfig):
        self.descriptions = load_descriptions(config, split)
        self.priming = config.priming
        self.num_train_examples = config.num_train_examples
        if self.priming:
            self.priming_prompt = build_priming_vqa_prompt(config, add_answer_period=True)
        self.rationale_generation = config.rationale_generation
        super().__init__(path, config)

    def process_single_item(self, item, **kwargs):
        image_id = item["question_id"]
        prompt = build_vqa_prompt(item, entities=self.descriptions["clip"], ocr_results=self.descriptions["ocr"],
                                  captions=self.descriptions["captions"])
        if self.rationale_generation:
            dataset = []
            for choice in item["choices"]:
                answer_prompt = prompt + " " + choice
                answer_prompt += ", because in this image,"
                dataset.extend(
                    super().process_single_item({"inputs_pretokenized": answer_prompt, "targets_pretokenized": "none"},
                                                image_id=image_id, choice=choice))
            return dataset
        else:
            if self.priming:
                prompt = self.priming_prompt + prompt
            if "direct_answers" in item:
                target = item["direct_answers"]
            else:
                target = "none"
            return super().process_single_item({"inputs_pretokenized": prompt, "targets_pretokenized": target},
                                               image_id=image_id, **kwargs)


class VQAGenTask(GenerationTask):
    config: VQAGenConfig

    @classmethod
    def config_class(cls):
        return VQAGenConfig

    def build_dataset(self, relative_path, split):
        return VQAGenDataset(os.path.join(self.config.path, relative_path), split, self.config)

    def save_prediction_to_file(self, file, predictions, data):
        tokenizer = get_tokenizer()
        results = {}
        for prediction, item in zip(predictions, data):
            prediction = tokenizer.detokenize(prediction)
            if self.config.priming or self.config.rationale_generation:
                prediction = prediction.split(".")[0]
            if self.config.rationale_generation:
                result = results.get(item["image_id"], {})
                result[item['choice']] = prediction
                results[item['image_id']] = result
            else:
                results[item["image_id"]] = {"direct_answer": prediction}
        with open("outputs/prediction_gen_" + datetime.now().strftime('%m-%d-%H-%M_') + file, "w") as output:
            json.dump(results, output)
