# -*- coding: utf-8 -*-
"""
Callable function that returns the Top-k probability distribution for the next character
"""
import json
import torch
from model import CharTransformer
from config import Config

class ProbabilityGenerator:
    def __init__(self, checkpoint_path):
        # Initialize model and vocabulary
        self.device = Config.device
        self.char2id, self.model = self._load_resources(checkpoint_path)
        self.id2char = {v: k for k, v in self.char2id.items()}
    
    def _load_resources(self, checkpoint_path):
        """Load model and vocabulary"""
        with open(Config.vocab_path, 'r', encoding='utf-8') as f:
            char2id = json.load(f)
        
        model = CharTransformer(len(char2id)).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])  # Extract 'model_state_dict'
        model.eval()  # Keep model in inference mode
        return char2id, model

    def get_topk_probabilities(self, prompt, topk=5):
        """
        Get Top-k probability distribution for the next character (pure data return)
        Args:
            prompt (str): Input text
            topk (int): Number of Top results to return (default 5)
        Returns:
            list: Top-k character probability list (elements are tuples: (character, probability))
            int: Number of valid characters in input (characters in vocabulary)
        """
        # Filter invalid characters (characters not in vocabulary)
        valid_chars = [c for c in prompt if c in self.char2id]
        if not valid_chars:
            return [], 0  # Return empty list and 0 when no valid characters
        
        # Convert input to tensor required by model
        input_ids = [self.char2id[c] for c in valid_chars]
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Model inference (no gradient mode for acceleration)
        with torch.no_grad():
            logits = self.model(input_tensor)[0, -1, :]  # Take output of last character
            probs = torch.softmax(logits, dim=-1)  # Convert to probability distribution
        
        # Get Top-k characters and probabilities (sorted by probability in descending order)
        topk_probs, topk_ids = torch.topk(probs, topk)
        results = [
            (self.id2char[idx.item()], prob.item())  # Convert to (character, probability) tuple
            for idx, prob in zip(topk_ids, topk_probs)
        ]

        return results, len(valid_chars)
        
    def get_full_probs(self, prompt):
        valid_chars = [c for c in prompt if c in self.char2id]
        if not valid_chars:
            return None
        input_ids = [self.char2id[c] for c in valid_chars]
        input_tensor = torch.tensor([input_ids], device=self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

# Usage example (demonstration of calling method only, no actual output)
if __name__ == "__main__":
    # Initialize generator (load model and vocabulary)
    generator = ProbabilityGenerator(
        checkpoint_path="/home/u2024000980/sanliwan/chartransformer_1/char_transformer_epoch9.pt"
    )

    # Input text
    input_text = "在務基本上就依托在農業社的基"

    # Call function to get Top-5 probabilities (no printing, only return data)
    top5_results, valid_len = generator.get_topk_probabilities(input_text, topk=5)

#Calling example
# from function import ProbabilityGenerator 

# def main():
#     # ------------------------- Step 1: Initialize generator -------------------------
#     # Model checkpoint path (replace with your actual path)
#     checkpoint_path = "/home/u2024000980/sanliwan/chartransformer_1/char_transformer_epoch4.pt"
#     generator = ProbabilityGenerator(checkpoint_path)

#     # ------------------------- Step 2: Call prediction function -------------------------
#     # Input text (replace with your actual input)
#     input_text = "在務基本上就依托在農業社的基"
#     # Call function to get Top-5 probability distribution (topk=5 is default, can be omitted)
#     top5_results, valid_len = generator.get_topk_probabilities(input_text, topk=5)

#     # ------------------------- Step 3: Process results -------------------------
#     if top5_results:  # Input is valid (at least one valid character exists)
#         print(f"Number of valid characters in input text: {valid_len}")
#         print("Top-5 character probability distribution:")
#         for char, prob in top5_results:
#             print(f"  Character: {char}, Probability: {prob:.6f}")
#     else:
#         print("Invalid input: No characters in text are in vocabulary")

# if __name__ == "__main__":
#     main()

