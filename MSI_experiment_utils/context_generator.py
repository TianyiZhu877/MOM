def needle_in_book(context_len, tokenizer, placement = 0.75) -> str:

    context = "A quick brown fox jumps over the lazy dog. \n"
    # with open("demo/duo_attention.txt", "r") as f:
    #     needle = f.read()
    needle = "Mary's favorite number is 34251"
    num_tokens_context = len(tokenizer.encode(context, add_special_tokens=False))
    num_repetitions = context_len // num_tokens_context

    placement = min(max(placement, 0), 1)
    
    text = (
        "This is a very long story book: <book> "
        + context * int(num_repetitions * placement)
        + needle
        + context * int(num_repetitions * (1 - placement))
        + "</book>\n Based on the content of the book, please briefly tell me about what is Mary's favorite number.\nAnswer:"
        # + "</book>\n Can you summarize this book?\nAnswer:"
    )

    return text


class needleInBook:
    def __init__(self, context_len, placement = 0.75, max_context = None):
        # if max_context is not None:
        #     context_len = min(max_context, context_len)
        self.context_len = context_len
        self.max_context = max_context

        self.placement = min(max(placement, 0), 1)


    def generate(self, tokenizer) -> str:

        context = "A quick brown fox jumps over the lazy dog. \n"
        needle = "Mary's favorite number is 34251"
        num_tokens_context = len(tokenizer.encode(context, add_special_tokens=False))
        num_repetitions = self.context_len // num_tokens_context

        
        text = ("This is a very long story book: <book> "
            + context * int(num_repetitions * self.placement)
            + needle
            + context * int(num_repetitions * (1 - self.placement))
        )

        if self.max_context is not None:
            cutoff = self.max_context/self.context_len * len(text)
            cutoff = min(max(int(cutoff), 0), len(text))
            text = text[:cutoff] 

        text = text + "</book>\n Based on the content of the book, please briefly tell me about what is Mary's favorite number.\nAnswer:"
        
        return text
    
    # def generate_query(self) -> str:
    #     return "</book>\n Based on the content of the book, please briefly tell me about what is Mary's favorite number.\nAnswer:"
    
    def evaluate(self, output:str) -> float:
        if "34251" in output:
            print("correct answer")
            return 1
        
        print("Incorrect answer")
        return 0
