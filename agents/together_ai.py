import os
import time
import requests
from types import SimpleNamespace
import together

class TogetherAIAgent():
    def __init__(self, kwargs: dict):
        self.api_key = together.api_key = os.getenv('TOGETHERAI_API_KEY')
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        self.args.model = "togethercomputer/" + self.args.model.removesuffix("-tg")

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "togethercomputer/llama-2-70b-chat"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0.0
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 0.95
        if not hasattr(self.args, 'repetition_penalty'):
            self.args.repetition_penalty = 1.0

    def generate(self, prompt):
        while True:
            try:
                output = together.Complete.create(
                    prompt = prompt, 
                    model = self.args.model, 
                    max_tokens = self.args.max_tokens,
                    temperature = self.args.temperature,
                    top_k = 1,
                    top_p = self.args.top_p,
                    repetition_penalty = 1.0,
                    stop = ['</s>']
                )
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.JSONDecodeError) as e:
                print("Error: {}\nRetrying...".format(e))
                time.sleep(2)
                continue

        return output

    def parse_basic_text(self, response):
        return response['output']['choices'][0]['text'].strip()

    def interact(self, prompt):
        while True:
            try:
                response = self.generate(prompt)
                output = self.parse_basic_text(response)
                break
            except:
                print("Error: Retrying...")
                time.sleep(2)
                continue

        return output
