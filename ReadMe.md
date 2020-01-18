# Condition Normalization CUDA extension

## Brief Introduction
This is a simple implemention for `Condition Normalization`(i.e. CN) with pytorch cuda extension.  

And it runs as fast as `Batch Normalization` in pytorch (3% slower exactly).  
Tests passed with pytorch 1.1.0 cuda 10.  

run following code to install the extension. (check cuda config first)
```bash
cd ./xcn_cuda
# python setup.py build
python setup.py install
```
then use `batch cn` or `instance cn` like this:
```python
import torch
from cond_norm import batchCondNorm, instCondNorm
x = torch.rand(4,3,10,10, requires_grad=True, device=torch.device('cuda'))
m = instCondNorm(3, minl=2.0).cuda()
y = m(x)
z = y.abs().sum()
z.backward()
```

Browse jupyter notebook in test folder for more details.


## Note:
`Condition Normalization` is a normalization layer like batch normalization, it runs as follows:
$$
y = w \cdot \frac{x-E(x)}{max(\sigma(x),l)} + b
$$
when set $l=0$, it is equal to `BN` with `track_running_stats=False`


