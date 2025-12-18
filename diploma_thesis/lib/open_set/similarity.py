import torch

def cosine_sim(a, b):

    return torch.nn.functional.cosine_similarity(a, b, dim = 0).item()