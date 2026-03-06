import logging
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from scripts.V5.configs.prompts import ULTIMATE_MCOT_PROMPT_TEMPLATE, ZERO_SHOT_PROMPT_TEMPLATE


class ReasoningEngine:
    def __init__(self, config: Dict[str, Any]):
        engine_config = config['model_configs']['reasoning_engine']
        model_path = engine_config['model_path']
        logging.info("正在初始化推理引擎...")
        logging.info(f" - 从 '{model_path}' 加载Qwen2.5-VL模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            load_in_8bit=engine_config.get('use_int8', False),
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if engine_config.get('use_flash_attention_2', False) else "eager",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        logging.info("✅ 推理引擎初始化完成。")

    def _parse_box_from_string(self, raw_output: str) -> Optional[List[int]]:
        match = re.search(r'(\d+[\.,]?\d*,\s*\d+[\.,]?\d*,\s*\d+[\.,]?\d*,\s*\d+[\.,]?\d*)', raw_output)
        if match:
            try:
                cleaned_str = match.group(1).replace('，', ',')
                return [int(float(n.strip())) for n in cleaned_str.split(',')]
            except (ValueError, IndexError):
                return None
        return None

    def _run_vlm(self, prompt: str, images: List[Image.Image]) -> Optional[List[int]]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)

        raw_output = self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        logging.info(f"--- [VLM原始输出] ---\n{raw_output}\n-------------------------")

        return self._parse_box_from_string(raw_output)

    def run_ultimate_mcot(self, image: Image.Image, text_prompt: str, reference_example: Dict) -> Optional[List[int]]:
        logging.info("执行带上下文的VLM推理 (MCoT)...")
        ref_image = Image.open(reference_example['data']['image_path']).convert("RGB")
        prompt = ULTIMATE_MCOT_PROMPT_TEMPLATE.format(
            text_prompt=text_prompt,
            rag_box=reference_example['data']['box']
        )
        return self._run_vlm(prompt, [image, ref_image])

    def run_zero_shot_vlm(self, image: Image.Image, text_prompt: str) -> Optional[List[int]]:
        logging.info("执行零样本VLM定位...")
        prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(text_prompt=text_prompt)
        return self._run_vlm(prompt, [image])