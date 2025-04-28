import argparse
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3, 4, 6, 7" # Restricting to use only GPU 6 and 7. > cuda:0 is 6 and cuda:1 is 7
import json
import re
from tqdm import tqdm
import shortuuid
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from CCOT_prompt import *
import torch
from torch.utils.data import Dataset, DataLoader
from pydantic import BaseModel, Field
from typing import TypeAlias, Optional, List

# os.environ["TOKENIZERS_PARALLELISM"] = "false" # disable tokenizer parallelism
base64str : TypeAlias = str
"""Data setting"""
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        ego_image_path = os.path.join(self.image_folder, line["image_folder"], "frame_aligned_videos", line["cam_ego"], line["frame"])
        exo_image_path = os.path.join(self.image_folder, line["image_folder"], "frame_aligned_videos", line["cam_exo"], line["frame"])
        question = line["question"]
        options = line["options"]
        answer = line["answer"]
        answer_ego = line["answer_ego"]
        answer_exo = line["answer_exo"]
        id = line["id"]
        frame = line["frame"]
        return {
            "ego_image_path": ego_image_path,
            "exo_image_path": exo_image_path,
            "question": question,
            "options": options,
            "answer": answer,
            "answer_ego": answer_ego,
            "answer_exo": answer_exo,
            "id": id,
            "perspective": line["perspective"],
            "cam_ego": line["cam_ego"],
            "cam_exo": line["cam_exo"],
            "image_folder": line["image_folder"],
            "frame": frame
        }
    def __len__(self):
        return len(self.questions)

def create_data_loader_default(questions, image_folder, batch_size=1, num_workers=1):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

"""VIC"""
class CCOT: #VIC gets question and image(list) as input
    def __init__(self, model_1, processor, device_1, datas, model_name, output_dir, question_file):
        self.model_1 = model_1
        self.processor = processor
        self.device_1 = device_1
        self.datas = datas
        self.model_name = model_name
        self.output_dir = output_dir
        self.question_file = question_file


    '''Async'''
    async def get_VLM_response(self, text_content:str, image_content:Optional[List[str]]=None, prompt:Optional[str]=None): #image need to be list or none/ prompt = system prompt
        if isinstance(prompt, str):#model_1
            content = []
            if isinstance(image_content, str):
                image_content = [image_content]
            for image in image_content:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": text_content})
            messages = [
                {"role": "system", "content": prompt},  # give prompt
                {"role": "user",
                 "content": content
                 }]
        else:
            content = []
            if isinstance(image_content, str):
                image_content = [image_content]
            for image in image_content:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": text_content})
            messages = [
                {"role": "user",
                 "content": content
                 }
            ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = self.device_1
        inputs = inputs.to(self.model_1.device)

        generated_ids = self.model_1.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        VLM_response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return VLM_response

    async def SG_generate(self, data):
        '''extracting data from data dictionary'''
        image1 = data['ego_image_path'][0]
        image2 = data['exo_image_path'][0]
        image = [image1, image2]
        data_question = data['question'][0]
        data_options_list = data['options']
        data_options = '\n'.join(f"{chr(65 + i)}) {option[0]}" for i, option in enumerate(data_options_list))
        question = data_question + '\n' + data_options
        '''SG generation process(1. ego/ 2. exo)'''
        VLM_system_prompt = system_prompt_ego
        VLM_question = question + sg_prompt
        ego_sg = await self.get_VLM_response(VLM_question, image[0], VLM_system_prompt) # ego json

        VLM_system_prompt = system_prompt_exo
        VLM_question = question + sg_prompt
        exo_sg = await self.get_VLM_response(VLM_question, image[1], VLM_system_prompt) # exo json
        '''Concat SG, use SG + 2image for question answering'''
        sg_data = f"Ego SG\n{ego_sg}\n\nExo SG\n{exo_sg}"
        final_question = sg_data + '\n' + question_prompt + '\n' + question #question_prompt has question presenting + answer format prompt
        VLM_system_prompt = system_prompt_both
        answer = await self.get_VLM_response(final_question, image, VLM_system_prompt)
        result = {"Answer": answer, "SG": sg_data}
        return [result, data]

    async def process_all_inputs(self): # dataloader loop 돌리기
        results=[]
        for data in tqdm(self.datas, total=len(self.datas)):
            result = await self.SG_generate(data)
            results.append(result)
            '''jsonl file saving / data format'''
            Result=result[0]
            model_answer = extract_answer_label(Result['Answer'])
            data=result[1]
            data_question = data['question'][0]
            data_options_list = data['options']
            data_options = '\n'.join(f"{chr(65 + i)}) {option[0]}" for i, option in enumerate(data_options_list))
            question = Result["SG"] + '\n' + question_prompt + '\n' + data_question + '\n' + data_options
            SG = Result['SG']
            letter = chr(65 + next(
                i for i, (item,) in enumerate(data_options_list) if item == data['answer'][0]))  # label 양식맞춰주기
            _answer_format = {
                "model_answer": f"[{model_answer}]",
                "label": f'{letter}) {data["answer"][0]}',
                "question_prompt": question,
                "Scene Graph" : SG,
                "model_name": self.model_name,
                "image_folder": data['image_folder'][0],
                "frame": data['frame'][0],
                "question": data_question,
                "options": re.findall(r'(?<=\)\s).*', data_options),
                "answer": data['answer'][0],
                "answer_ego": data['answer_ego'][0],
                "answer_exo": data['answer_exo'][0],
                "perspective": data['perspective'][0],
                "cam_ego": data['cam_ego'][0],
                "cam_exo": data['cam_exo'][0],
                "id": data['id'][0],
                "modle_answer_saved": model_answer
            }
            with open(os.path.join(self.output_dir, self.question_file), "a") as f:
                json.dump(_answer_format, f)
                f.write("\n")
        return results


'''answer formatting'''###Not used in CCOT
def extract_answer_label(answer):
    if isinstance(answer, list) and len(answer) > 0:
        answer_str = answer[0]
        match = re.search(r'([A-D])', answer_str)
        if match:
            label= match.group(1)
            return label
    if isinstance(answer, str) and len(answer) > 0:
        match = re.search(r'([A-D])', answer)
        if match:
            label= match.group(1)
            return f"[{label}]"
    return "Invalid answer format\nCould be small letters"

"""최종 모델 돌리기"""
def eval_model(args):
    '''seed setting'''
    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    '''model initialization - Using only one model'''
    device_1=torch.device("cuda:0")
    model_1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map=device_1).eval().to(device_1)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    '''arg setting'''
    model_name = args.model#for output directory, for representation just using the VLM as model_name
    image_dir = args.image_dir
    question_dir = args.question_dir
    output_dir = os.path.join(args.output_dir, model_name)  # /sdc1/datasets/EEH_v2/model_output/jyjang + model_name
    os.makedirs(output_dir, exist_ok=True)
    question_files = [f for f in os.listdir(question_dir) if f.endswith(".jsonl")] # human_filtered_multi 파일안의 json파일을 spatial_xo.json, spatial_ego ...
    # finished_files = ['spatial_exo.jsonl'] # 오류 발생 시 빼고 돌릴 수 있도록
    # question_files = [file for file in question_files if file not in finished_files]
    for question_file in question_files:
        questions = []
        with open(os.path.join(question_dir, question_file), "r") as f:
            print(f"processing question: {question_file}")
            for line in f:
                line = line.strip()
                data_dict = json.loads(line)
                questions.append(data_dict)
        questions = questions[:100]
        datas=create_data_loader_default(questions, image_dir, batch_size=1, num_workers=1)
        '''DDCOT Class Initialization'''
        CCOT_instance = CCOT(model_1, processor, device_1, datas, model_name, output_dir, question_file)
        '''DDCOT using async'''
        loop = asyncio.get_event_loop()
        Result = loop.run_until_complete(CCOT_instance.process_all_inputs())

"""argparse, __main__"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2_5_VL_CCOT")
    parser.add_argument("--image-dir", type=str, default="/sdc1/datasets/EEH_v2/test/")
    parser.add_argument("--output-dir", type=str, default="/sdc1/datasets/EEH_v2/model_output/jyjang")
    parser.add_argument("--question-dir", type=str, default="/sdc1/datasets/EEH_v2/outputs_final/human_filtered_multi")
    args = parser.parse_args()
    eval_model(args)