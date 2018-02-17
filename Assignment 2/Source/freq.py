import numpy as np
S = np.random.randint(0, 5, 20)
print("List:")
print(S)
print("Most frequent in the List:")
print(np.bincount(S).argmax())