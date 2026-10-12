# head scoring function
def score(pattern, rows, offset):
    cols = rows - offset
    pattern = pattern[:, :, rows, cols]
    scores = pattern.mean(dim=(0, 2))

    return scores

