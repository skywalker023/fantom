import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

class HuggingFaceAgent():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_text(self, text):
        return text

    def postprocess_output(self, response):
        return response

    def interact(self, text):
        prompt = self.preprocess_text(text)
        encoded_texts = self.tokenizer(prompt, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = self.postprocess_output(decoded_output)

        return response

    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        encoded_texts = self.tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]

        return responses

class FlanT5Agent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = T5Tokenizer.from_pretrained("google/" + args.model)
        self.model = T5ForConditionalGeneration.from_pretrained("google/" + args.model, device_map="auto")

class FlanUL2Agent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)

class MistralAIAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        if 'instruct' in self.args.model.lower():
            model_name = "Mistral-7B-Instruct-v0.1"
        else:
            model_name = "Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/" + model_name)
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/" + model_name, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token 

    def preprocess_text(self, text):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)

    def postprocess_output(self, response):
        return response.split("[/INST]")[-1].strip()

class ZephyrAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/" + self.args.model)
        self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/" + self.args.model, device_map="auto")

    def preprocess_text(self, text):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)

    def postprocess_output(self, response):
        return response.split("\n<|assistant|>\n")[-1].strip()
