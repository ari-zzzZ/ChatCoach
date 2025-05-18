import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftMixedModel

class QwenChatbot:
    def __init__(self, 
                 base_model_path: str,
                 adapter_a_path: str,
                 adapter_b_path: str,
                 system_prompt: str = "你是一位心理咨询师。"):
        #  加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        #  加载基座模型
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True
        )

        #  用 PeftMixedModel 同时加载两个 LoRA adapter
        self.model = PeftMixedModel.from_pretrained(base, adapter_a_path)
        self.model.load_adapter(adapter_b_path, adapter_name="stage2")
        # 激活两套 adapter
        self.model.set_adapter(["default", "stage2"])
        self.model.eval()

        # 4) 初始化对话历史
        self.history = [{"role": "system", "content": system_prompt}]
    
    def generate(self, user_input: str, enable_thinking: bool = True) -> str:
        self.history.append({"role": "user", "content": user_input})
        text = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        #gen = self.model.generate(**inputs, max_new_tokens=512)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=512, use_cache=True)
        #  prompt 部分
        out_ids = gen[0][len(inputs.input_ids[0]):]
        reply = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        self.history.append({"role": "assistant", "content": reply})
        return reply

if __name__ == "__main__":
    bot = QwenChatbot(
        base_model_path="./SFTuned",
        adapter_a_path="./DPO_1",
        adapter_b_path="./DPO_2",
    )
    print("=== Qwen3 Chat (按 exit/quit 结束) ===")
    while True:
        q = input("You: ")
        if q.strip().lower() in ("exit", "quit"):
            print("Bye!")
            break
        ans = bot.generate(q, enable_thinking=False)
        print("Bot:", ans)