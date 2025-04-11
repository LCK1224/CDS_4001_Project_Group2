import random


def mask_text():
    # random.seed(1234)
    tokens = text.split()
    mask_prob = 0.2
    it = iter([random.uniform(0, 5) for _ in range(10)])
    masked_tokens = []

    for token in tokens:
        x = next(it, 0.0)
        if x < mask_prob:
            print(x)
            masked_tokens.append(" ")
        else:
            masked_tokens.append(token)
    return masked_tokens


text = 'I go to school by bus'


print(mask_text())
print(mask_text())
print(mask_text())
