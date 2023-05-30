# Karpathy Lectures nn-zero-to-hero([link](https://github.com/karpathy/nn-zero-to-hero)) with complete exercises.




## Lecture 1: Micrograd
Micrograd is a tiny Autograd engine that implements backpropagation (reverse-mode autodiff),you will learn the fundamentals of automatic differentiation by building a scalar value autograd engine with only one class Value and using it to be able to create a small MLP.

Code Example:
```python

#input
x1=Value(2.0,label='x1')
# weight w1
w1 = Value(-3.0, label='w1')
#bias
b = Value(6.8813735870195432, label='b')
#forward pass
x1w1=x1*w1;x1w1.label='x1*w1'
n=x1w1+b; n.label='n'

#activation function
o=n.tanh();o.label='o'
#backward pass
o.backward()
#draw graph
draw_dot(o)
```
![Graph](graph.png)
## Lecture 2: Bigram 
Bigram is a language model and its an n-gram model for n=2,we are looking at a previous char to predict the next one .

 







