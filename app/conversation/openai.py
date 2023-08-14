import openai
import json
import logging
import re
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIConversation():
    def __init__(self,):
        self.directives = self.load_system_directives()
        self.dialogue_stack = []
    
    def set_api_key(self, api_key):
        openai.api_key = api_key
    
    def load_system_directives(self):
        '''Load the json file containing system directives'''
        # print the cwd
        print(os.getcwd())
        with open('app/conversation/directives/system-directives.json') as f:
            system_directives = json.load(f)
        return system_directives

    async def chatgpt_response(self, prompt, model):
        current_prompt = {"role": "user", "content": prompt}
        self.dialogue_stack.append(current_prompt)
        self.dialogue_stack = self.dialogue_stack[-6:]
        messages = self.directives + self.dialogue_stack

        logger.info("MODEL: " + model)
        logger.info("PROMPT: " + prompt.lower())
        try:
            response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=1800,
            )
            response_content = response['choices'][0]['message']['content']
            logger.info("RESPONSE: " + response_content)
            re.compile(r'As an .* AI')
            if not re.search(r'As an .* AI', response_content):
                self.dialogue_stack.append({"role": "assistant", "content": response_content })
            return response_content
        except Exception as e:
            logger.info("OPENAI Error: " + str(e))
            return "I'm sorry, I'm having trouble understanding you. Could you try rephrasing that?"       