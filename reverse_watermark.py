from models.watermark_faster import watermark_model, binary_encoding_function
import warnings

import numpy as np

# Ignore warnings
warnings.filterwarnings('ignore')


class clear_watermark(watermark_model):
    def __init__(self, language, mode, tau_word, lamda):
        super().__init__(language, mode, tau_word, lamda)
        self.rng = np.random.default_rng(12345)

    def get_candidate_encodings(self, tokens, enhanced_candidates, index_space):
        best_candidates = [candidates[:2][-1] for candidates, _ in enhanced_candidates]
        new_index_space = index_space
        
        # for init_candidates, masked_token_index in zip(enhanced_candidates, index_space):
        #     filtered_candidates = []
            
        #     for idx, candidate in enumerate(init_candidates):
        #         if masked_token_index-1 in new_index_space:
        #             bit = binary_encoding_function(best_candidates[-1]+candidate[0])
        #         else:
        #             bit = binary_encoding_function(tokens[masked_token_index-1]+candidate[0])
                
        #         if 0==bit:
        #             filtered_candidates.append(candidate)

        #     # Sort the candidates based on their scores
        #     filtered_candidates = sorted(filtered_candidates, key=lambda x: x[1], reverse=True)

        #     if len(filtered_candidates) >= 1: 
        #         best_candidates.append(filtered_candidates[0][0])
        #         new_index_space.append(masked_token_index)

        return best_candidates, new_index_space


tau_word = 0.8
lammy = 0.8
mode = "embed"

model = watermark_model(language="English", mode=mode, tau_word=tau_word, lamda=lammy)
clear_model = clear_watermark(language="English", mode=mode, tau_word=tau_word, lamda=lammy)

text = ("Flocking is a type of coordinated group behavior that is exhibited by animals of various species, including birds, "
"fish, and insects. It is characterized by the ability of the animals to move together in a coordinated and cohesive "
"manner, as if they were a single entity. Flocking behavior is thought to have evolved as a way for animals to "
"increase their chances of survival by working together as a group. For example, flocking birds may be able to locate "
"food more efficiently or defend themselves against predators more effectively when they work together."
)

watermarked_text = model.embed(text)


is_watermark, p_value, n, ones, z_value = model.watermark_detector_fast(watermarked_text)

print("===========================")
print(f"Original Text:")
print(text)
print("---")
print(f"Watermarked Text:")
print(watermarked_text)
print(f"p_value: {p_value}, n: {n}, ones: {ones}, z_value: {z_value}, confidence: {(1 - p_value) * 100:.2f}%")


# watermarked_text = "Flocking is a kind of coordinated team behavior that is displayed by birds of several species , notably birds , fish , and insects . It is characterized by the ability of the organisms to move together in a coordinated and cohesive way , as if they were a single entity .Flocking behavior is suggested to have evolved as a way for animals to expand their likelihood of survival by acting together as a group . For instance , flocking birds could be able to locate food more efficiently or defend themselves against prey more effectively when they work together ."

unwatermarked_text = clear_model.embed(watermarked_text)


is_watermark, p_value, n, ones, z_value = model.watermark_detector_fast(unwatermarked_text)

print("===========================")

print(f"Unwatermarked Text:")
print(unwatermarked_text)

print(f"p_value: {p_value}, n: {n}, ones: {ones}, z_value: {z_value}, confidence: {(1 - p_value) * 100:.2f}%")
