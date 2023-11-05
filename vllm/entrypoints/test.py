from vllm.entrypoints.llm import LLM, SamplingParams
from vllm.model_executor.adapters import lora

# Create an LLM.
llm = LLM(model="/home/hercules/llama-2-hf/Llama-2-7b-hf/")

# Add LoRA adapter
lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, "/home/hercules/llama-2-hf/Llama-2-7b-16k-sft")

prompts = [
    "I like to",
    "Brazil is",
    "Faça uma redação de 100 palavras sobre o Brasil.",
]

sampling_params = SamplingParams(temperature=0, top_k=-1)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    