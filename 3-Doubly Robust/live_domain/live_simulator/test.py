import numpy as np
from scipy.stats import multinomial

a = np.random.dirichlet(np.ones(5))
b = np.random.multinomial(1, a)
print(b)
print(b.argmax())
