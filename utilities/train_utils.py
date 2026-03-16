import math
import numpy as np
import hashlib
from einops import rearrange, repeat

def cosine_schedule(start_val, end_val, total_epochs, warmup_epochs=0):
    def schedule(epoch):
        if epoch < warmup_epochs:
            return start_val + (end_val - start_val) * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return end_val + (start_val - end_val) * cosine_decay
    return schedule

def linear_schedule(start_val, end_val, total_epochs):
    def schedule(epoch):
        progress = epoch / total_epochs
        return start_val + (end_val - start_val) * progress
    return schedule

def get_seed_from_id(subject_id):
    # Hash the subject ID
    hashed_id = hashlib.sha256(subject_id.encode()).hexdigest()
    # Convert the hash to an integer and use it as the seed
    seed = int(hashed_id, 16) % (2**32)  # Ensure the seed fits in 32 bits
    return seed

