import os
import json
import re
import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from verl.utils.reward_score.ttrl_math.math_normalize import normalize_answer as math_normalize
from verl.utils.reward_score.ttrl_math.math_utils import (
    last_boxed_only_string,
    remove_boxed,
    grade_answer_mathd,
    grade_answer_sympy,
)

# ================= 配置区域 =================
# 1. 基座模型路径 (用于加载 Tokenizer/模型)
BASE_MODEL_PATH = "/HOME/HDD_POOL/Qwen2.5-VL-7B"

# 2. LoRA adapter 路径（使用基座模型 + LoRA 推理）
LORA_PATH = "/HOME/HDD_POOL/ttrl_vision/verl/checkpoints/TTRL-verl/geoQA-Qwen2.5-VL-7B/0108/TTRL-Len@3k-grpo-163636/global_step_654/actor/lora_adapter"
LORA_NAME = "ttrl_lora"
LORA_INT_ID = 1

# 3. 测试集路径
DATA_PATH = "/HOME/HDD_POOL/ttrl_vision/verl/data/MathVision/test.json"

# 4. 输出文件
OUTPUT_FILE = "inference_results_math_vision.json"

# 5. 推理参数配置
INFERENCE_CONFIG = {
    "num_samples": 1,          # 每个问题生成的样本数 (Pass@16)
    "max_tokens": 2048,         # 最大生成长度
    "temperature": 0.7,         # 温度系数
    "top_p": 0.9,               # Nucleus sampling
    "top_k": 50,                # Top-k sampling
    "repetition_penalty": 1.0,  # 重复惩罚
}

# 6. vLLM 系统配置
SYSTEM_CONFIG = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "enforce_eager": False
}

# 7. 调试与输出设置
VERBOSE = True                  # 是否打印详细的生成结果
SHOW_FULL_RESPONSE = False      # 是否打印完整的回复文本 (如果 False 只打印提取的答案)
USE_IMAGES = True               # 是否包含图像（多模态数据集）
# ===========================================

def extract_answer(text):
    if text is None:
        return ""
    text = text.strip()
    if not text:
        return ""
    boxed = last_boxed_only_string(text)
    if boxed:
        unboxed = remove_boxed(boxed)
        if unboxed is not None:
            return unboxed.strip()
    lowered = text.lower()
    if lowered.startswith("yes"):
        return "Yes"
    if lowered.startswith("no"):
        return "No"
    m = re.match(r"^\s*([A-D])[\.\)]?\s*$", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return ""

def normalize_answer(answer):
    """Normalize an answer string using the same logic as math_normalize."""
    if answer is None:
        return ""
    return math_normalize(str(answer))


def check_correctness(pred, gold):
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    # 忽略大小写（用于 A/B/C/D 选项等简单字符串）
    if pred_norm.lower() == gold_norm.lower():
        return True

    if isinstance(pred, str) and isinstance(gold, str):
        pred_letter = pred.strip().upper()
        gold_str = gold.strip()
        if pred_letter in {"A", "B", "C", "D"}:
            gold_head = gold_str[:1].upper()
            if gold_head == pred_letter:
                return True
        if pred_norm:
            gold_numbers = re.findall(r"-?\d+\.?\d*", gold_str)
            if any(pred_norm == normalize_answer(num) for num in gold_numbers):
                return True

    # 更严格的数学等价判断
    try:
        if grade_answer_mathd(pred_norm, gold_norm):
            return True
    except Exception:
        pass

    try:
        if grade_answer_sympy(pred_norm, gold_norm):
            return True
    except Exception:
        pass

    return False

def main():
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    with open(DATA_PATH, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    
    prompts = [d['prompt'] for d in data]
    golds = [d['answer'] for d in data]
    print(f"Loaded {len(prompts)} examples.")

    print(f"Loading tokenizer from {BASE_MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    processor = None
    if USE_IMAGES:
        print(f"Loading processor from {BASE_MODEL_PATH}...")
        try:
            processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading processor: {e}")
            return

    # 预处理 Prompts (应用 Chat Template)
    print("Formatting prompts with chat template...")
    formatted_prompts = []
    image_inputs = []
    data_dir = os.path.dirname(os.path.abspath(DATA_PATH))
    project_root = os.path.abspath(os.path.join(data_dir, "..", ".."))
    def ensure_min_image_size(image, min_size=28):
        width, height = image.size
        if width < min_size or height < min_size:
            width = max(width, min_size)
            height = max(height, min_size)
            return image.resize((width, height), Image.BICUBIC)
        return image

    for prompt, item in zip(prompts, data):
        images = None
        if USE_IMAGES:
            img_entries = item.get("images") or []
            images = []
            for img in img_entries:
                if isinstance(img, dict) and img.get("path"):
                    img_path = img["path"]
                    if isinstance(img_path, str) and img_path.startswith("file://"):
                        img_path = img_path[7:]
                    if not os.path.isabs(img_path):
                        if img_path.startswith("data/"):
                            img_path = os.path.join(project_root, img_path)
                        else:
                            img_path = os.path.join(data_dir, img_path)
                    img = Image.open(img_path).convert("RGB")
                    img = ensure_min_image_size(img)
                    images.append(img)

        system_msg = {"role": "system", "content": "Please directly answer the question and provide the correct option letter and  put your answer within \\boxed{}."}
   
        if processor is not None and images:
            content_parts = [{"type": "image"} for _ in images]
            content_parts.append({"type": "text", "text": prompt})
            user_msg = {"role": "user", "content": content_parts}
            messages = [system_msg, user_msg]
        else:
            messages = [
                system_msg,
                {"role": "user", "content": prompt},
            ]
        if processor is not None:
            text_input = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        formatted_prompts.append(text_input)
        image_inputs.append(images)

    print(f"Initializing vLLM with base model: {BASE_MODEL_PATH}...")
    try:
        llm = LLM(
            model=BASE_MODEL_PATH,
            tokenizer=BASE_MODEL_PATH,
            tensor_parallel_size=SYSTEM_CONFIG["tensor_parallel_size"],
            trust_remote_code=True,
            gpu_memory_utilization=SYSTEM_CONFIG["gpu_memory_utilization"],
            enforce_eager=SYSTEM_CONFIG["enforce_eager"],
            enable_lora=True,
            max_lora_rank=128,
        )
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        return

    sampling_params = SamplingParams(
        n=INFERENCE_CONFIG["num_samples"],
        temperature=INFERENCE_CONFIG["temperature"],
        top_p=INFERENCE_CONFIG["top_p"],
        top_k=INFERENCE_CONFIG["top_k"],
        max_tokens=INFERENCE_CONFIG["max_tokens"],
        repetition_penalty=INFERENCE_CONFIG["repetition_penalty"],
    )

    lora_request = None
    if LORA_PATH:
        lora_request = LoRARequest(
            lora_name=LORA_NAME,
            lora_int_id=LORA_INT_ID,
            lora_path=LORA_PATH,
        )

    print("Starting generation...")
    if USE_IMAGES:
        vllm_inputs = []
        for text, images in zip(formatted_prompts, image_inputs):
            entry = {"prompt": text}
            if images:
                entry["multi_modal_data"] = {"image": images}
            vllm_inputs.append(entry)
        outputs = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)

    print("Evaluating results...")
    detailed_results = []
    pass_16_list = []

    for i, output in enumerate(outputs):
        gold = golds[i]
        prompt = prompts[i]
        
        generated_texts = [o.text for o in output.outputs]
        sample_results = []

        if VERBOSE:
            print(f"Processing {i+1}/{len(prompts)}...")
            print(f"  Question: {prompt[:100]}..." if len(prompt) > 100 else f"  Question: {prompt}")
            print(f"  Gold Answer: {gold}")

        correct_count = 0
        for idx, text in enumerate(generated_texts):
            pred = extract_answer(text)
            is_correct = check_correctness(pred, gold)
            if is_correct:
                correct_count += 1

            if VERBOSE and (idx == 0 or SHOW_FULL_RESPONSE):
                print(f"  [Sample {idx+1}] Pred: {pred} | Correct: {is_correct}")
                if SHOW_FULL_RESPONSE:
                    print(f"  Response: {text[:200]}...")

            sample_results.append({
                "text": text,
                "extracted": pred,
                "correct": is_correct
            })

        p16 = 1.0 if correct_count > 0 else 0.0
        pass_16_list.append(p16)

        if VERBOSE:
            print(f"  Result: Pass@16={p16:.2f}")
            print("-" * 20)
        
        detailed_results.append({
            "id": i,
            "prompt": prompt,
            "gold": gold,
            "pass@16": pass_16_list[-1],
            "samples": sample_results
        })

    avg_pass_16 = np.mean(pass_16_list)

    print("="*40)
    print("Evaluation Complete.")
    print(f"Pass@1 : {avg_pass_16:.4%}")
    print("="*40)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
