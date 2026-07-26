class SimpleLLM:
    def __init__(self):
        # Simulate a very small "knowledge base" or "trained patterns".
        # In a real LLM, these patterns are learned from vast amounts of text data.
        # Keys represent input sequences, values are the predicted next word.
        self.knowledge = {
            "merhaba": "nasılsın",
            "nasılsın": "iyiyim",
            "bugün hava": "güzel",
            "yapay zeka": "ilginç",
            "büyük dil": "modelleri",
            "karpathy": "llm",
            "chatgpt nasıl": "çalışır",
            "bu bir": "örnek",
            "örnek": "metin",
            "llm": "nedir",
            "nedir": "bir",
            "bir": "yapay",
            "yapay": "zeka",
            "zeka": "modelidir"
        }

    def tokenize(self, text):
        """
        A very basic tokenizer that splits text into words and converts to lowercase.
        Real LLMs use more sophisticated tokenizers (e.g., BPE, WordPiece) to handle
        punctuation and sub-word units more effectively.
        """
        return text.lower().replace(",", "").replace(".", "").split()

    def predict_next_word(self, prompt_words):
        """
        Predicts the next word based on the last word(s) in the prompt.
        This is a highly simplified lookup, not a statistical or neural network model.
        It demonstrates the core idea of an LLM: given context, predict the next token.
        """
        if not prompt_words:
            return None

        # Try to match the last two words for more context (a very basic form of attention/context window)
        if len(prompt_words) >= 2:
            two_word_key = " ".join(prompt_words[-2:])
            if two_word_key in self.knowledge:
                return self.knowledge[two_word_key]

        # Fallback to matching just the last word if a two-word context isn't found
        last_word = prompt_words[-1]
        if last_word in self.knowledge:
            return self.knowledge[last_word]

        return None # Cannot predict if no pattern matches

    def generate_response(self, prompt, max_length=10):
        """
        Generates a response by iteratively predicting the next word.
        This loop simulates how an LLM generates text token by token.
        """
        tokens = self.tokenize(prompt)
        generated_tokens = list(tokens) # Start with the prompt tokens

        print(f"Prompt: '{prompt}'")
        print(f"Initial tokens: {generated_tokens}")

        for i in range(max_length - len(tokens)):
            next_word = self.predict_next_word(generated_tokens)
            if next_word:
                generated_tokens.append(next_word)
                print(f"  Step {i+1}: Predicted '{next_word}'. Current sequence: {generated_tokens}")
            else:
                print(f"  Step {i+1}: No further prediction possible.")
                break # Stop if no prediction can be made

        # Reconstruct the sentence, capitalizing the first word for readability
        response = " ".join(generated_tokens)
        if response:
            response = response[0].upper() + response[1:]
        return response

# --- Example Usage ---
if __name__ == "__main__":
    llm = SimpleLLM()

    print("\n--- Next Word Prediction Examples ---")
    print(f"Predict after 'merhaba': {llm.predict_next_word(llm.tokenize('merhaba'))}")
    print(f"Predict after 'büyük dil': {llm.predict_next_word(llm.tokenize('büyük dil'))}")
    print(f"Predict after 'chatgpt nasıl': {llm.predict_next_word(llm.tokenize('chatgpt nasıl'))}")
    print(f"Predict after 'bilgisayar': {llm.predict_next_word(llm.tokenize('bilgisayar'))} (No match)")

    print("\n--- Response Generation Examples ---")
    print(f"\nGenerated response for 'merhaba':")
    print(f"  -> {llm.generate_response('merhaba')}")

    print(f"\nGenerated response for 'bugün hava':")
    print(f"  -> {llm.generate_response('bugün hava')}")

    print(f"\nGenerated response for 'karpathy':")
    print(f"  -> {llm.generate_response('karpathy')}")

    print(f"\nGenerated response for 'yapay zeka':")
    print(f"  -> {llm.generate_response('yapay zeka')}")

    print(f"\nGenerated response for 'büyük dil':")
    print(f"  -> {llm.generate_response('büyük dil')}")

    print(f"\nGenerated response for 'bilgisayar':")
    print(f"  -> {llm.generate_response('bilgisayar')}") # Will only echo prompt as no prediction is possible

    print(f"\nGenerated response for 'chatgpt nasıl':")
    print(f"  -> {llm.generate_response('chatgpt nasıl')}")
