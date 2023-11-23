import hashlib
import re
from models.watermark_faster import watermark_model
import warnings

import numpy as np

# Ignore warnings
warnings.filterwarnings('ignore')


class secret_watermark(watermark_model):
    def __init__(self, secret, language, mode, tau_word, lamda):
        super().__init__(language, mode, tau_word, lamda)
        self.secret = secret

    def binary_encoding_function(self, token):
        hash_value = int(hashlib.sha256((token + self.secret).encode('utf-8')).hexdigest(), 16)
        random_bit = hash_value % 2
        return random_bit

class clear_watermark(watermark_model):
    def __init__(self, language, mode, tau_word, lamda):
        super().__init__(language, mode, tau_word, lamda)
        self.rng = np.random.default_rng(12345)

    def watermark_embed(self,text):
        input_text = text
        # Tokenize the input text
        tokens = self.tokenizer.tokenize(input_text) 
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        masked_tokens=tokens.copy()
        start_index = 1
        end_index = len(tokens) - 1
        
        index_space = []
       
        for masked_token_index in range(start_index+1, end_index-1):
            if not self.pos_filter(tokens,masked_token_index,input_text):
                continue
            index_space.append(masked_token_index)
        
        if len(index_space)==0:
            return text
        init_candidates, new_index_space = self.candidates_gen(tokens,index_space,input_text, 8, 0)
        if len(new_index_space)==0:
            return text
        
        enhanced_candidates, new_index_space = self.filter_candidates(init_candidates,tokens,new_index_space,input_text)
        
        enhanced_candidates, new_index_space = self.get_candidate_encodings(tokens, enhanced_candidates, new_index_space)
        
        for init_candidate, masked_token_index in zip(enhanced_candidates, new_index_space):
            tokens[masked_token_index] = init_candidate

        watermarked_text = self.tokenizer.convert_tokens_to_string(tokens[1:-1])
    
        if self.language == 'Chinese':
            watermarked_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])', '', watermarked_text)
        return watermarked_text

    def get_candidate_encodings(self, tokens, enhanced_candidates, index_space):
        best_candidates = [candidates[:self.rng.integers(1,3)][-1][0] for candidates in enhanced_candidates]
        new_index_space = index_space

        return best_candidates, new_index_space


tau_word = 0.8
lammy = 0.8
mode = "embed"

model = secret_watermark("ubc-cpen442", language="English", mode=mode, tau_word=tau_word, lamda=lammy)
clear_model = clear_watermark(language="English", mode=mode, tau_word=tau_word, lamda=lammy)

text = ("Flocking is a type of coordinated group behavior that is exhibited by animals of various species, including birds, "
"fish, and insects. It is characterized by the ability of the animals to move together in a coordinated and cohesive "
"manner, as if they were a single entity. Flocking behavior is thought to have evolved as a way for animals to "
"increase their chances of survival by working together as a group. For example, flocking birds may be able to locate "
"food more efficiently or defend themselves against predators more effectively when they work together."
)


is_watermark, p_value, n, ones, z_value = model.watermark_detector_precise(text)

print("===========================")
print(f"Original Text:")
print(text)
print(f"FROM ORIGINAL MODEL: p_value: {p_value}, n: {n}, ones: {ones}, z_value: {z_value}, confidence: {(1 - p_value) * 100:.2f}%")
print("---")

watermarked_text = model.embed(text)

is_watermark, p_value, n, ones, z_value = model.watermark_detector_precise(watermarked_text)

print(f"Watermarked Text:")
print(watermarked_text)
print(f"FROM ORIGINAL MODEL: p_value: {p_value}, n: {n}, ones: {ones}, z_value: {z_value}, confidence: {(1 - p_value) * 100:.2f}%")

is_watermark, p_value, n, ones, z_value = clear_model.watermark_detector_precise(watermarked_text)

print(f"FROM CLEAR MODEL   : p_value: {p_value}, n: {n}, ones: {ones}, z_value: {z_value}, confidence: {(1 - p_value) * 100:.2f}%")


# watermarked_text = "Flocking is a kind of coordinated team behavior that is displayed by birds of several species , notably birds , fish , and insects . It is characterized by the ability of the organisms to move together in a coordinated and cohesive way , as if they were a single entity .Flocking behavior is suggested to have evolved as a way for animals to expand their likelihood of survival by acting together as a group . For instance , flocking birds could be able to locate food more efficiently or defend themselves against prey more effectively when they work together ."
# print("===========================")
# print(f"Original Text:")
# print(text)
# print("---")
# print(f"Watermarked Text:")
# print(watermarked_text)
# print("p_value: 0.0005075004735565336, n: 30, ones: 24, z_value: 3.2863353450309964, confidence: 99.95%")



unwatermarked_text = clear_model.embed(watermarked_text)

print("===========================")

print(f"Unwatermarked Text:")
print(unwatermarked_text)

is_watermark, p_value, n, ones, z_value = model.watermark_detector_precise(unwatermarked_text)

print(f"FROM ORIGINAL MODEL: p_value: {p_value}, n: {n}, ones: {ones}, z_value: {z_value}, confidence: {(1 - p_value) * 100:.2f}%")
