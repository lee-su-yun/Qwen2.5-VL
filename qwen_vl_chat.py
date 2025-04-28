import os
import json
import asyncio
from typing import List, Optional
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # Qwen2.5-VL util

class RobotTaskPlanner:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    async def generate_plan(self, task: str, images: List[str], system_prompt: Optional[str] = None) -> str:
        # Prepare message
        content = []
        for image in images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": task})

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

        # Prepare model input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def parse_generated_text(self, text: str) -> List[dict]:
        """Parse the model output text into a list of subtasks."""
        subtasks = []
        lines = text.split('\n')
        current = {}

        for line in lines:
            line = line.strip()
            if line.lower().startswith("subtask:"):
                if current:
                    subtasks.append(current)
                    current = {}
                current["subtask_description"] = line[len("Subtask:"):].strip()
            elif line.lower().startswith("potential issue:"):
                current["potential_issue"] = line[len("Potential Issue:"):].strip()
            elif line.lower().startswith("solution:"):
                current["solution"] = line[len("Solution:"):].strip()

        if current:
            subtasks.append(current)

        return subtasks

async def create_robot_plan_and_save(
    model, processor, device,
    task: str,
    image_paths: List[str],
    image_ids: List[int],
    output_json_path: str
):
    planner = RobotTaskPlanner(model, processor, device)

    system_prompt = (
        "You are a robotics expert. Based on the provided images and task, "
        "break down the task into small, executable subtasks for a one-armed robot. "
        "For each subtask, identify potential unexpected issues and suggest practical solutions. "
        "Use the following format clearly:\n\n"
        "Subtask: <description>\n"
        "Potential Issue: <issue>\n"
        "Solution: <solution>\n"
    )

    # 1. Generate response
    generated_text = await planner.generate_plan(task, image_paths, system_prompt)

    # 2. Parse into structured format
    structured_subtasks = planner.parse_generated_text(generated_text)

    # 3. Assemble final output
    output = {
        "task": task,
        "images_used": image_paths,
        "subtasks": structured_subtasks
    }

    # 4. Save JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Saved robot plan to {output_json_path}")

if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # Example input
    image_paths = [
        "/home/sylee/codes/Qwen2.5-VL/images/stack_cups1.jpg",
        "/home/sylee/codes/Qwen2.5-VL/images/stack_cups2.jpg",
        "/home/sylee/codes/Qwen2.5-VL/images/stack_cups3.jpg"
    ]
    image_ids = [1, 2, 3]  # or actual file names if you prefer
    task = "On the desk, stack the cups by nesting them in the following sequence: place the red cup first, then the blue cup inside it, and finally the purple cup on top." #"Clean up the desk."

    output_json_path = "/home/sylee/codes/Qwen2.5-VL/outputs/stack_cups.json"

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        create_robot_plan_and_save(
            model,
            processor,
            device,
            task,
            image_paths,
            image_ids,
            output_json_path
        )
    )