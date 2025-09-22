import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastchat.model import get_conversation_template
from typing import List, Optional, Tuple, Union
from tqdm import tqdm


class EEG(nn.Module):
    def __init__(self, model_path, prototype_path, configs=None):
        super().__init__()

        self.model_path = model_path
        if model_path == 'meta-llama/Llama-2-7b-chat-hf':
            self.template_name = 'llama-2'
        elif model_path  == 'lmsys/vicuna-7b-v1.5':
            self.template_name = 'vicuna'
        elif model_path == 'TheBloke/guanaco-7B-HF':
            self.template_name = 'guanaco'
        self.model, self.tokenizer = self.load_model(model_path)
        self.conv_template = self.load_conversation_template(self.template_name)
        self.generation_config = self.model.generation_config

        if configs is not None:
            self.generation_config.update(**configs)

        self.emb_pos, self.emb_neg = self.load_prototype(prototype_path)
        

    def load_conversation_template(self, template_name):
        conv_template = get_conversation_template(template_name)
        if conv_template.name == 'zero_shot':
            conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
            conv_template.sep = '\n'
        elif conv_template.name == 'llama-2':
            conv_template.sep2 = conv_template.sep2.strip()

        return conv_template


    
    def load_model(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer
    

    def get_prompt(self, prompt_str, target=None):
        self.conv_template.messages=[]
        self.conv_template.append_message(self.conv_template.roles[0], f"{prompt_str}")
        self.conv_template.append_message(self.conv_template.roles[1], target)
        prompt = self.conv_template.get_prompt()
        return prompt

    def get_input_ids(self, prompt_str, target=None):
        prompt = self.get_prompt(prompt_str, target)
        toks = self.tokenizer(prompt).input_ids
        return torch.tensor(np.array([toks])).cuda()

    
    def generate(self, input_prompts=None, input_ids=None):
        if input_prompts is not None:
            input_ids = self.get_input_ids(input_prompts)
        
        output_ids = self.model.generate(input_ids, return_dict_in_generate=False, generation_config = self.generation_config)[0]
        # print(input_ids.shape)
        generated_output = output_ids[input_ids.shape[1]:-1]
        return self.tokenizer.decode(generated_output)


    def get_next_token_embedding(self, input_prompts=None, input_ids=None):
        if input_prompts is not None:
            input_ids = self.get_input_ids(input_prompts)
        
        embeddings = self.model.forward(input_ids)
        return embeddings.hidden_states[:, -1, :].clone()


    def detokenize(self, ids):
        return self.tokenizer.decode(ids)
        

    def execute_with_layer_number(self, num_layers, input_prompts=None, input_ids=None):
        assert num_layers <= len(self.model.model.layers)

        layer_temp = self.model.model.layers
        self.model.model.layers = layer_temp[:num_layers]
        output = self.generate(input_prompts, input_ids)
        self.model.model.layers = layer_temp
        return output


    def update_generation_config(self, configs):
        self.generation_config.update(**configs)

    
    def layer_wise_embedding(self, input_prompts=None, input_ids=None):
        self.model.model.config.output_hidden_states = True

        if input_prompts is not None:
            input_ids = self.get_input_ids(input_prompts)
        with torch.no_grad():
            embeddings = self.model.forward(input_ids) 
        self.model.model.config.output_hidden_states = False
        result = [np.squeeze(item[:, -1, :].cpu().numpy()) for item in embeddings.hidden_states]
        return np.array(result)


    def safe_generate(self, score_limit, alpha, input_prompts=None, input_ids=None):
        safety_score = self.eeg_classifier(alpha, input_prompts, input_ids)
        if safety_score<score_limit:
            return self.generate(input_prompts, input_ids), safety_score
        else:
            return "Sorry, but I can not assist with that.", safety_score


    def eeg_classifier(self, alpha=0.75, input_prompts=None, input_ids=None):
        emb = self.layer_wise_embedding(input_prompts, input_ids)
        score = 0
        emb_pos, emb_neg = self.emb_pos.numpy(), self.emb_neg.numpy()
        
        max_curr = 0
        curr_exit = 0
        result_list = []
        
        for i in range(1, int(32*alpha)+1):
            cos_sim_normalized_neg = np.dot(emb[i] / np.linalg.norm(emb[i]), emb_neg[i] / np.linalg.norm(emb_neg[i]))
            cos_sim_normalized_pos = np.dot(emb[i] / np.linalg.norm(emb[i]), emb_pos[i] / np.linalg.norm(emb_pos[i]))
            if cos_sim_normalized_neg < cos_sim_normalized_pos:
                score += 1
        return score

    
    def layer_wise_attention(self, input_prompts=None, input_ids=None):
        self.model.model.config.output_attentions = True

        if input_prompts is not None:
            input_ids = self.get_input_ids(input_prompts)
        with torch.no_grad():
            embeddings = self.model.forward(input_ids)
        self.model.model.config.output_attentions = False
        return embeddings.attentions

    def set_gcg_system_prompt(self):
        self.conv_template.system_message = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    

    def calculate_prompt_prototype(self, prompt, label):
        prompt_embed = []
        for p in tqdm(prompt):
            prompt_embed.append(self.layer_wise_embedding(p))
        emb_pos = []
        emb_neg = []
        prompt_embed = np.array(prompt_embed)
        label = np.array(label)
        for i in range(33):
            curr_layer_emb_pos = np.zeros(4096)
            curr_layer_emb_neg = np.zeros(4096)
            for j in range(prompt_embed.shape[0]):
                if label[j] == 0:
                    curr_layer_emb_neg += prompt_embed[j][i] / np.linalg.norm(prompt_embed[j][i])
                else:
                    curr_layer_emb_pos += prompt_embed[j][i] / np.linalg.norm(prompt_embed[j][i])
            emb_pos.append(curr_layer_emb_pos*np.mean(label))
            emb_neg.append(curr_layer_emb_neg*(1-np.mean(label)))
        return emb_pos, emb_neg

    def save_prototype(self, emb_pos, emb_neg, file_path):
        proto = torch.tensor([emb_pos, emb_neg])
        torch.save(proto, file_path)

    def load_prototype(self, file_path):
        prototype = torch.load(file_path)
        return prototype[0], prototype[1]
