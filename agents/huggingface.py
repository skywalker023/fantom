import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration


class HuggingFaceAgent():
    def __init__(self, args):
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
            output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = self.postprocess_output(decoded_output)

        return response

    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        encoded_texts = self.tokenizer(batch_prompts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]

        return responses

class FlanT5Agent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = T5Tokenizer.from_pretrained("google/" + args.model)
        self.model = T5ForConditionalGeneration.from_pretrained("google/" + args.model, device_map="auto")

    def interact(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = self.postprocess_output(decoded_output)

        return response

class FlanUL2Agent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)

class MistralAIAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token 

    def preprocess_text(self, text):
        chat = [
          {"role": "user", "content": text},
        ]
        return chat

    def batch_preprocess_text(self, text):
        return "<s>[INST]" + text + "[/INST]"

    def postprocess_output(self, response):
        return response.removeprefix("[/INST] ")

    def interact(self, text):
        prompt = self.preprocess_text(text)
        encoded_texts = self.tokenizer.apply_chat_template(prompt, return_tensors="pt")
        input_ids = encoded_texts.to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128)
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        response = self.postprocess_output(decoded_output[0].split(text)[-1].strip())

        return response

    def batch_interact(self, batch_texts):
        batch_prompts = [self.batch_preprocess_text(text) for text in batch_texts]
        encoded_texts = self.tokenizer(batch_prompts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output.split(batch_texts[idx])[-1].strip()) for idx, decoded_output in enumerate(decoded_outputs)]

        return responses

class ZephyrAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
        self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", device_map="auto")

