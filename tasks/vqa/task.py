import os
import json
import re
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


def build_vqa_prompt(item, descriptions, predict_rationale=False):
    image_id = item["image_id"]
    question_id = item["question_id"]
    overall_sent = f'Please observe the image and answer the question.'
    entities = descriptions["clip"]
    ocr_results = descriptions["ocr"]
    captions = descriptions["captions"]
    supports = descriptions["supports"]
    rationales = descriptions["rationales"]
    object_entities = descriptions["object_clip"]
    object_captions = descriptions["object_captions"]
    caption_sent = None
    if captions is not None:
        caption = captions[image_id]["caption"]
        caption_sent = f"This image is {caption}"
    scene_sent, entity_sent = None, None
    if entities is not None:
        description = entities[image_id]
        img_type, sorted_places = description["img_type"], description["sorted_places"]
        entity_list = description["object_list"]
        entity_result = ", ".join(entity_list)
        scene_sent = f'I think this image was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.'
        entity_sent = f'I think there might be a {entity_result} in this {img_type}.'
    ocr_sent = None
    if ocr_results is not None:
        ocr_result = ocr_results[image_id]
        if ocr_result["texts"]:
            texts = sorted(ocr_result["texts"], key=lambda x: x["confidence"], reverse=True)
            texts = ", ".join([f'"{item["text"]}"' for item in texts[:5]])
            ocr_sent = f'The texts in the image include {texts}.'
    object_sent = None
    if object_entities is not None or object_captions is not None:
        object_sents = []
        for o in item["objects"]:
            object_id = f"{image_id}/{o}"
            object_sent = []
            if object_entities is not None:
                obj_entity_list = object_entities[object_id]["object_list"]
                obj_attribute_list = object_entities[object_id]["attribute_list"]
                obj_entity_result = ", ".join(obj_entity_list[:3])
                obj_attribute_result = ", ".join(obj_attribute_list[:3])
                object_sent.append(f"{o} might be a {obj_entity_result}")
                object_sent.append(f'{o} might be {obj_attribute_result}')
            if object_captions is not None:
                obj_caption = object_captions[object_id]['caption'][0]
                object_sent.append(f'{o} might be {obj_caption}')
            object_sent = ". ".join(object_sent)
            object_sents.append(object_sent)
        object_sent = ". ".join(object_sents) + "."
    support_sent = None
    if supports is not None:
        support_sents = supports[question_id]
        if support_sents:
            support_sent = ". ".join(support_sents) + "."
    prompts = [overall_sent]
    if caption_sent is not None:
        prompts.append(caption_sent)
    if scene_sent is not None:
        prompts.append(scene_sent)
    if ocr_sent is not None:
        prompts.append(ocr_sent)
    if entity_sent is not None:
        prompts.append(entity_sent)
    if object_sent is not None:
        prompts.append(object_sent)
    if support_sent is not None:
        prompts.append(support_sent)
    prompt = " ".join(prompts)
    question = item["question"]
    prompt = prompt + question
    if predict_rationale:
        prompt = prompt + " Because"
    elif rationales is not None:
        rationale = rationales[question_id].strip(".")
        prompt = prompt + " Because " + rationale[0].lower() + rationale[1:] + ", the answer is"
    else:
        prompt = prompt + " Answer:"
    return prompt


def filter_support(support, topk=5, threshold=0.8):
    support_sents = []
    sentences, values = [], []
    for choice, matching_result in support.items():
        for sent, value in matching_result:
            sentences.append(sent)
            values.append(value)
    values = torch.tensor(values)
    top_values, top_indices = torch.topk(values, k=min(len(sentences), topk), dim=-1)
    for value, idx in zip(reversed(top_values.tolist()), reversed(top_indices.tolist())):
        if value > threshold:
            support_sents.append(sentences[idx])
    return support_sents


def load_descriptions(config, split):
    descriptions = {"clip": None, "ocr": None, "captions": None, "supports": None, "rationales": None,
                    "object_clip": None, "object_captions": None}
    if config.clip_pattern is not None:
        clip_file_path = os.path.join(config.description_path, config.clip_pattern[split])
        descriptions["clip"] = load_description_file(clip_file_path)
        print_rank_0(f"Loading {split} clip descriptions from {clip_file_path}")
    if config.ocr_pattern is not None:
        ocr_file_path = os.path.join(config.description_path, config.ocr_pattern[split])
        descriptions["ocr"] = load_description_file(ocr_file_path)
        print_rank_0(f"Loading {split} OCR results from {ocr_file_path}")
    if config.caption_pattern is not None:
        caption_file_path = os.path.join(config.description_path, config.caption_pattern[split])
        descriptions["captions"] = load_description_file(caption_file_path)
        print_rank_0(f"Loading {split} captions from {caption_file_path}")
    if config.object_clip_pattern is not None:
        object_clip_file_path = os.path.join(config.description_path, config.object_clip_pattern[split])
        descriptions["object_clip"] = load_description_file(object_clip_file_path)
        print_rank_0(f"Loading {split} object clip descriptions from {object_clip_file_path}")
    if config.object_caption_pattern is not None:
        object_caption_file_path = os.path.join(config.description_path, config.object_caption_pattern[split])
        descriptions["object_captions"] = load_description_file(object_caption_file_path)
        print_rank_0(f"Loading {split} object caption descriptions from {object_caption_file_path}")
    if config.support_pattern is not None:
        support_file_path = os.path.join(config.description_path, config.support_pattern[split])
        with open(support_file_path) as file:
            supports = json.load(file)
        supports = {
            image_id: filter_support(support, topk=config.support_topk, threshold=config.support_threshold) for
            image_id, support in supports.items()}
        descriptions['supports'] = supports
        print_rank_0(f"Loading {split} supports from {support_file_path}")
    if config.rationale_pattern is not None and split in config.rationale_pattern:
        rationale_file_path = os.path.join(config.description_path, config.rationale_pattern[split])
        with open(rationale_file_path) as file:
            rationales = json.load(file)
        rationales = {key: value["direct_answer"][0] for key, value in rationales.items()}
        descriptions['rationales'] = rationales
        print_rank_0(f"Loading {split} rationales from {rationale_file_path}")
    return descriptions


def read_dataset(path):
    if not path.endswith("jsonl"):
        try:
            with open(os.path.join(path), "r", encoding="utf-8") as file:
                dataset = json.load(file)
            return dataset
        except json.decoder.JSONDecodeError:
            pass
    dataset = []
    with open(os.path.join(path), "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            dataset.append(item)
    return dataset


def build_priming_vqa_prompt(config, add_rationale=False, cot_rationale=False):
    priming_prompt = ""
    assert config.train_path is not None
    dataset = read_dataset(config.train_path)
    descriptions = load_descriptions(config, "train")
    descriptions["rationales"] = None
    if cot_rationale:
        descriptions["rationales"] = {item["question_id"]: item["rationales"][0] for item in dataset}
    for item in random.sample(dataset, config.num_train_examples):
        item = process_vqa_item(item, dataset_name=config.name)
        prompt = build_vqa_prompt(item, descriptions)
        answer = item["choices"][item["correct_choice_idx"]]
        priming_prompt += prompt + " " + answer.strip(".")
        if add_rationale:
            priming_prompt += ", because in this image, " + random.choice(item["rationales"]).strip(".")
        priming_prompt += ". "
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
    description_path: str = None
    clip_pattern: Union[str, Dict[str, str]] = None  # Organize data file in groups
    ocr_pattern: Union[str, Dict[str, str]] = None
    caption_pattern: Union[str, Dict[str, str]] = None
    object_clip_pattern: Union[str, Dict[str, str]] = None
    object_caption_pattern: Union[str, Dict[str, str]] = None
    support_pattern: Union[str, Dict[str, str]] = None
    rationale_pattern: Union[str, Dict[str, str]] = None
    priming: bool = False
    num_train_examples: int = 10
    train_path: str = None
    support_topk: int = 5
    support_threshold: float = 0.8
    replace_object: bool = False


@dataclass
class VQAMulConfig(VQAConfig, MultiChoiceTaskConfig):
    pass


@dataclass
class VQAGenConfig(VQAConfig, GenerationTaskConfig):
    rationale_generation: bool = False
    cot_rationale: bool = False
    pass


def process_vqa_item(item, dataset_name="aokvqa"):
    if "a-okvqa" in dataset_name:
        return item
    elif "vcr" in dataset_name:
        def vcr_detokenizer(string):
            string = (
                string.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
            )
            return string
        def process_tokenized_input(words):
            if not isinstance(words, str):
                words = [" and ".join(list(map(str, word))) if isinstance(word, list) else word for word in words]
                words = " ".join(words)
            words = vcr_detokenizer(words)
            return words

        new_item = {"image_id": item["img_id"], "question_id": item["annot_id"],
                    "correct_choice_idx": item["answer_label"]}
        question = process_tokenized_input(item["question"])
        new_item["question"] = question
        choices = [process_tokenized_input(choice) for choice in item["answer_choices"]]
        new_item["choices"] = choices
        objects = set()
        for word in item["question"]:
            if isinstance(word, list):
                objects.update(word)
        for choice in item["answer_choices"]:
            for word in choice:
                if isinstance(word, list):
                    objects.update(word)
        new_item["objects"] = list(objects)
        if "rationale" in item:
            new_item["rationales"] = [item["rationale"]]
        return new_item
    else:
        raise NotImplementedError


def replace_object_indices(text, objects):
    obj_sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, obj in enumerate(objects):
        if i < len(obj_sequence):
            text = re.sub(r"\b" + str(obj), obj_sequence[i], text)
            text = re.sub(r"\." + str(obj), "." + obj_sequence[i], text)
    return text


class VQAMulDataset(MultiChoiceTaskDataset):
    config: VQAMulConfig

    def __init__(self, path, split, config: VQAMulConfig):
        self.descriptions = load_descriptions(config, split)
        self.priming = config.priming
        self.num_train_examples = config.num_train_examples
        if self.priming:
            self.priming_prompt = build_priming_vqa_prompt(config, cot_rationale=config.rationale_pattern is not None)
        super().__init__(path, config)

    def process_single_item(self, item, **kwargs):
        item = process_vqa_item(item, dataset_name=self.config.name)
        image_id = item["question_id"]
        prompt = build_vqa_prompt(item, self.descriptions)
        if self.priming:
            prompt = self.priming_prompt + prompt
        choices = item["choices"]
        if self.config.replace_object:
            prompt = replace_object_indices(prompt, item["objects"])
            choices = [replace_object_indices(choice, item["objects"]) for choice in choices]
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
            if "a-okvqa" in self.config.name:
                results[item["image_id"]] = {"multiple_choice": item["choices_pretokenized"][prediction]}
            elif "vcr" in self.config.name:
                results[item["image_id"]] = {"multiple_choice": prediction}
            else:
                raise NotImplementedError(self.config.name)
        file_name = file.split(".")[0]
        with open("outputs/prediction_mul_" + datetime.now().strftime(
                '%m-%d-%H-%M_') + self.config.name + "_" + file_name + ".json", "w") as output:
            json.dump(results, output)


class VQAGenDataset(GenerationTaskDataset):
    config: VQAGenConfig

    def __init__(self, path, split, config: VQAGenConfig):
        self.descriptions = load_descriptions(config, split)
        self.priming = config.priming
        self.rationale_generation = config.rationale_generation
        self.cot_rationale = config.cot_rationale
        self.num_train_examples = config.num_train_examples
        if self.priming:
            self.priming_prompt = build_priming_vqa_prompt(config, add_rationale=self.rationale_generation,
                                                           cot_rationale=self.cot_rationale or config.rationale_pattern is not None)
        super().__init__(path, config)

    def process_single_item(self, item, **kwargs):
        item = process_vqa_item(item, dataset_name=self.config.name)
        image_id = item["question_id"]
        prompt = build_vqa_prompt(item, self.descriptions, predict_rationale=self.cot_rationale)
        if self.rationale_generation:
            dataset = []
            for choice in item["choices"]:
                answer_prompt = prompt + " " + choice
                answer_prompt += ", because in this image,"
                if self.priming:
                    answer_prompt = self.priming_prompt + answer_prompt
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
            if self.config.return_all_beams:
                prediction = [tokenizer.detokenize(p) for p in prediction]
            else:
                prediction = tokenizer.detokenize(prediction)
            if self.config.rationale_generation:
                result = results.get(item["image_id"], {})
                result[item['choice']] = prediction
                results[item['image_id']] = result
            else:
                results[item["image_id"]] = {"direct_answer": prediction}
        if self.config.rationale_generation:
            prefix = "prediction_rationale_"
        else:
            prefix = "prediction_gen_"
        file_name = file.split(".")[0]
        with open(os.path.join("outputs", prefix + datetime.now().strftime(
                '%m-%d-%H-%M_') + self.config.name + "_" + file_name + ".json"), "w") as output:
            json.dump(results, output)
