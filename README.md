# Bayesian Optimization Library

BayesTuner optimizes hyperparameters for machine learning algorithms.  It is suited for all sorts of problems.  
The library is still in its early stage of development.

To install:

```
pip install -i https://test.pypi.org/simple/ bayestuner
```

**Example:**

```python
import numpy as np
from bayestuner.tuner import BayesTuner

tuner = BayesTuner(objective = lambda X : - X[0]**2 - X[1]**2,
                    bounds = [(-5.12,5.12,'continuous')]*2,
                    n_iter = 60,
                    init_samples = 15)

res = tuner.tune(print_score = True)      #Summary of the optimization
argmax = res.x
maximum = res.func_value
```

You can also optimize on discrete and continuous domains:

```python
bounds = [(-5.12,5.12,'continuous'),(-10,10,'discrete')]
```
