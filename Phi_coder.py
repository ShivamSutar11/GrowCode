import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# ---------------------------
# 1. Model + device setup
# ---------------------------

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

if torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float16
    print("✅ Using Apple GPU (MPS)")
else:
    device = torch.device("cpu")
    dtype = torch.float32
    print("⚠️ MPS not available, using CPU")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Phi-3 mini model (should use cache, no big download now)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
)
model.to(device)
model.eval()
print("Model ready ✅")

# ---------------------------
# 2. Inference
# ---------------------------

def generate_answer(
    instruction: str,
    user_input: str,
    max_new_tokens: int = 1024,  # big budget so it can finish
) -> str:
    """
    instruction: what kind of help you want
    user_input: problem statement + code
    """
    system_prompt = (
        "You are a friendly DSA coding tutor. "
        "Explain the full idea clearly, step-by-step."
    )

    prompt = (
        f"<system>{system_prompt}</system>\n"
        f"<user>{instruction}\n\n{user_input}</user>\n"
        f"<assistant>"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,   # up to 512 new tokens
            do_sample=False,                 # greedy = stable explanations
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if "<assistant>" in full_text:
        answer = full_text.split("<assistant>")[-1]
    else:
        answer = full_text

    return answer.strip()


# ---------------------------
# 3. Gradio UI – GrowCode
# ---------------------------

HELP_TYPES = [
    "Explain the solution",
    "Find the bug and explain",
    "Give hint and solution",
    "Optimize the code",
]

# how detailed each mode should be
MAX_TOKENS_BY_MODE = {
    "Explain the solution": 180,        # detailed explanation
    "Find the bug and explain": 180,    # detailed
    "Optimize the code": 120,           # medium
    "Give only a hint": 48,             # short & fast
}

def coding_helper_ui(help_type, problem_and_code):
    if not problem_and_code.strip():
        return "Please paste a problem statement and/or code."

    instruction = help_type
    user_input = problem_and_code

    # Always give the model the full budget (512 tokens)
    return generate_answer(instruction, user_input, max_new_tokens=1024)



def main():
    with gr.Blocks() as demo:
        gr.Markdown("## 🌱🚀 GrowCode – DSA Coding Helper")
        gr.Markdown(
            "Welcome to **GrowCode**! Paste your problem + code, choose how you want help, "
            "and I’ll explain, debug, or optimize your solution step-by-step."
        )

        with gr.Row():
            help_type = gr.Dropdown(
                HELP_TYPES,
                value="Find the bug and explain",
                label="Type of help",
            )

        problem_and_code = gr.Textbox(
            lines=14,
            label="Problem statement + code",
            placeholder="Paste your DSA problem and/or code here...",
        )

        ask_btn = gr.Button("Ask GrowCode")

        answer_box = gr.Textbox(
            lines=14,
            label="Model answer",
        )

        ask_btn.click(
            fn=coding_helper_ui,
            inputs=[help_type, problem_and_code],
            outputs=[answer_box],
        )

    demo.launch()

if __name__ == "__main__":
    main()
