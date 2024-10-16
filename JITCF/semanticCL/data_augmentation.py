import random
import torch


def random_deletion(tokens, p=0.1):
    if len(tokens) == 1:
        return tokens

    new_tokens = []
    for token in tokens:
        r = random.uniform(0, 1)
        if r > p:
            new_tokens.append(token)

    if len(new_tokens) == 0:
        return [random.choice(tokens)]
    else:
        return new_tokens


def random_swap(tokens, n=1):
    tokens = tokens.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(tokens)), 2)
        tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
    return tokens


def random_replacement(tokens, p=0.1):
    new_tokens = tokens.copy()
    for i, token in enumerate(tokens):
        r = random.uniform(0, 1)
        if r < p:
            new_tokens[i] = random.choice(tokens)
    return new_tokens


def augment(inputs):
    augmented_inputs = []
    for input_ids in inputs:
        input_ids = input_ids.tolist()
        choice = random.choice(['deletion', 'swap', 'replacement'])
        if choice == 'deletion':
            aug_input_ids = random_deletion(input_ids)
        elif choice == 'swap':
            aug_input_ids = random_swap(input_ids)
        elif choice == 'replacement':
            aug_input_ids = random_replacement(input_ids)
        else:
            aug_input_ids = input_ids

        if len(aug_input_ids) > len(input_ids):
            aug_input_ids = aug_input_ids[:len(input_ids)]
        else:
            aug_input_ids += [0] * (len(input_ids) - len(aug_input_ids))

        augmented_inputs.append(torch.tensor(aug_input_ids))
    return torch.stack(augmented_inputs)
