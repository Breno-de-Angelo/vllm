import openai
# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"


def read_txt_file(material_txt):
    content = ""
    with open(material_txt) as f:
        for line in f.readlines():
            content += line
    return content

prompt = "Instruction:\n"
# prompt += "The day has been tiring. I woke up, had breakfast, took the bus, worked and played basketball." 
prompt += read_txt_file("brazil_wiki_short.txt")
# prompt += read_txt_file("hills_like_white_elephants.txt")
prompt += "\n\nQuestion:"
prompt += "\nSummarize the text above."
completion = openai.Completion.create(model="../../llama-2-hf/Llama-2-7b-hf/",
                                      prompt=prompt,
                                      max_tokens=512)
print("Completion result:", completion)