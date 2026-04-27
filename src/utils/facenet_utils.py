import numpy as np

SIMILARITY_THRESHOLD = 0.7


def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> tuple[bool, float]:
    """
    Compare two 512-float FaceNet embeddings using cosine similarity.
    Returns (match, distance) where distance is in [0, 2] (lower = more similar).
    """
    e1 = embedding1.flatten().astype(np.float32)
    e2 = embedding2.flatten().astype(np.float32)

    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)

    if norm1 == 0 or norm2 == 0:
        return False, 2.0

    cosine_sim = np.dot(e1, e2) / (norm1 * norm2)
    distance = 1.0 - float(cosine_sim)

    match = distance < (1.0 - SIMILARITY_THRESHOLD)
    return match, distance
