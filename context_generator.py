def needle_in_book(context_len, tokenizer) -> str:

    context = "A quick brown fox jumps over the lazy dog. \n"
    # with open("demo/duo_attention.txt", "r") as f:
    #     needle = f.read()
    needle = "Mary's favorite number is 34251"
    num_tokens_context = len(tokenizer.encode(context, add_special_tokens=False))
    num_repetitions = context_len // num_tokens_context

    text = (
        "This is a very long story book: <book> "
        + context * int(num_repetitions * 0.75)
        + needle
        + context * int(num_repetitions * (1 - 0.75))
        + "</book>\n Based on the content of the book, please briefly tell me about what is Mary's favorite number.\nAnswer:"
        # + "</book>\n Can you summarize this book?\nAnswer:"
    )

    return text
