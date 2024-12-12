import google.generativeai as palm
import google.ai.generativelanguage as gen_lang
import time

from .Model import Model


class PaLM2(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        self.api_keys = api_keys
        api_pos = int(config["api_key_info"]["api_key_use"])
        if api_pos == -1: # use all keys
            self.key_id = 0
            self.api_key = None
        else: # only use one key at the same time
            assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
            self.api_key = api_keys[api_pos]
            self.set_API_key()
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        
    def set_API_key(self):
        palm.configure(api_key=self.api_key)
        
    def query(self, msg):
        if self.api_key == None:
            palm.configure(api_key=self.api_keys[self.key_id % len(self.api_keys)])
            self.key_id += 1
            
        try:
            completion = palm.generate_text(
                model=self.name,
                prompt=msg,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                safety_settings=[
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_DEROGATORY,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_TOXICITY,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_VIOLENCE,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_SEXUAL,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_MEDICAL,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_DANGEROUS,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                ]
            )
            response = completion.result

        except Exception as e:
            print(e)
            if 'not supported' in str(e):
                return ''
            else:
                print('Error occurs! Please check output carefully.')
                time.sleep(300)
                return self.query(msg)
        
        if response == '' or response == None:
            response = 'Input may contain harmful content and was blocked by PaLM.'

        return response