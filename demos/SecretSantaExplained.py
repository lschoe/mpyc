#!/usr/bin/env python
# coding: utf-8

# # Secret Santa explained
# 
# Step by step, we develop a secure MPC protocol for the [Secret Santa](https://en.wikipedia.org/wiki/Secret_Santa) problem. Traditionally, a group of family and friends gathers to put all their names in a hat and then randomly draw names from the hat. If someone draws their own name, they have to start all over again. 
# 
# Mathematically, the Secret Santa problem is about generating so-called [derangements](https://en.wikipedia.org/wiki/Derangement), which are permutations without fixed points. A permutation is a one-to-one mapping on a set of numbers, and a fixed-point is a number that is mapped to itself. 
# 
# We present an MPyC program (a Python program using the `mpyc` package) for generating uniformly random derangements. 
# In this notebook, the MPyC program is run by a single party only. However, this very same MPyC program can be run between multiple parties to generate secret-shared random derangements, as will be shown at the end. These random derangements will remain secret *forever* unless a majority of these parties collude.
# 
# ## MPyC setup
# 
# To get started, we simply import the MPyC runtime `mpc` at the start of our program.

# In[1]:


from mpyc.runtime import mpc


# A derangement of length $n$ is a permutation of the numbers $0, 1, ..., n-1$ without fixed points. 
# 
# Represented as Python lists, the 12 smallest derangements are:
# 
# |$n$| <p align="justify">length-$n$ derangements|
# |---|:----------------------|
# | 2 | <p align="left">[1,0]  |
# | 3 | <p align="left">[1,2,0], [2,0,1]  |
# | 4 | <p align="left">[1,0,3,2], [1,2,3,0], [1,3,0,2], [2,0,3,1], [2,3,0,1], [2,3,1,0], [3,0,1,2], [3,2,0,1], [3,2,1,0]  |
# 
# To represent *secret-shared* derangements, we will use a secure MPyC type of integers. For simplicity, we choose 32-bit (default) secure integers.

# In[2]:


secint = mpc.SecInt() # 32-bit secure MPyC integers


# Finally, we start the `mpc` runtime, which means that point-to-point connections between each pair of parties will be established.

# In[3]:


mpc.run(mpc.start())  # required only when run with multiple parties 


# Insertion of this call ensures that the Python code can also be run with multiple parties, as shown at the end of this notebook.

# #### _Quick note on MPyC coroutines_
# 
# All computations pertaining to secure integers of type `secint` are done asynchronously in MPyC. The implementation of MPyC builds heavily on Python's `asyncio` module. Secure computations in MPyC are in fact implemented as MPyC coroutines, which are a special type of Python coroutines.
# 
# For example, consider the following code fragment.

# In[4]:


a = secint(5)    
b = secint(13)
c, d = a + b, a - b                   # c, d are placeholders of type secint
e = c * d                             # e is a placeholder of type secint
f = mpc.run(mpc.output(e))            # forces actual computation of c, d, e
print('Example: (5+13)*(5-13) =', f) 


# Evaluation of this piece of code first creates three (empty) placeholders of type `secint`. The last but one line forces the actual computation of the (secret-shared) values of `c`, `d`, and `e`.
# 
# In this notebook, however, we will not focus on the asynchronous aspects of MPyC. Readers not familiar with coroutines and asynchronous computation may therefore skip to the next section, ignoring all aspects of asynchronous computation. Most MPyC code is meant to be understandable without bothering about the (order of) execution of the code anyway!
# 
# A Python coroutine, defined by the keyword `async` at the start of a function definition, is turned into an MPyC coroutine by using the decorator `mpc.coroutine`.
# When called, an MPyC coroutine will return immediately (nonblocking). The main difference with Python coroutines is that an MPyC coroutine will return a placeholder or, more generally, nested lists/tuples containing placeholders. The placeholders are typed (e.g., of type `secint`, or any other secure MPyC type), and the type of the placeholders is defined by the first `await` expression in the MPyC coroutine, using the `mpc.returnType` method. 

# ## Random derangements from random permutations
# 
# To generate a uniformly random derangement, we basically proceed in the traditional way: we randomly draw all numbers from a hat and start all over if there are any fixed points. A permutation $p$, viewed as a sequence of length $n$, is a derangement exactly when $\forall_{0\leq i<n} \ p_i \neq i$.
# 
# We represent a secret derangement $p$ by a list with elements of type `secint`. The idea is to first generate a random permutation and then check if this permutation happens to be free of fixed points. If any fixed points are found, we start all over again.
# 
# To use this idea for a secure computation, we need a way to check if $p$ is free of fixed points without leaking any further information about $p$. The elements of $p$ are all secret and should remain so. The following property tells us how we can find this single bit of information on $p$ by a simple computation over secure integers:
# 
# $$\forall_{0\leq i<n} \ p_i \neq i \ \ \ \Leftrightarrow \ \ \ \prod_{i=0}^{n-1} \ (p_i - i) \neq 0$$
# 
# The leads to the following MPyC code for function `random_derangement`.

# In[5]:


@mpc.coroutine                                    # turn coroutine into an MPyC coroutine
async def random_derangement(n):                  # returns list of n secint elements
    await mpc.returnType(secint, n)               # set return type of this MPyC coroutine
    p = random_permutation(n)
    t = mpc.prod([p[i] - i for i in range(n)])    # securely multiply all differences p[i] - i
    if await mpc.is_zero_public(t):               # publicly test whether t is equal to zero
        return random_derangement(n)              # recurse if t is zero
    else:
        return p                                  # done if t is nonzero


# Function `random_derangement` uses these four functions from the `mpc` runtime:
# 
# 1. `mpc.prod(x)` to securely compute the product of all elements in `x`.
# 
# 2. `mpc.is_zero_public(a)` to test securely whether `a` is equal to 0, revealing only the outcome publicly.
# 
# 3. Decorator `mpc.coroutine(f)` to turn coroutine `f` into an MPyC coroutine.
# 
# 4. `mpc.returnType(rettype)` to define the return type of an MPyC coroutine.
# 
# We have defined function `random_derangement` as a coroutine because from its body we call function `mpc.is_zero_public`, which is also a coroutine, and we need its the result to follow the correct branch of the `if` statement. The execution of `random_derangement` is suspended at the `await` keyword, and will be resumed once the result `mpc.is_zero_public(t)` is available.
# 
# ## Random permutations from random unit vectors
# 
# The [Fisher-Yates shuffle (or, Knuth shuffle)](https://en.wikipedia.org/wiki/Fisher–Yates_shuffle) is a classic algorithm for generating permutations uniformly at random. The Python program is very simple:
# 
# ```
#     p = list(range(n))
#     for i in range(n-1):
#         r = random.randrange(i, n)
#         p[i], p[r] = p[r], p[i]
# ```
# 
# Each of the $n!$ permutations is generated with probability $1/n!$, corresponding with the fact that variable `r` takes on uniformly random values, first among $n$ values, then $n-1$ values, $n-2$ values, ..., down to $2$ values at the end.
# 
# To implement a random shuffle securely, we have to hide which elements of `p` are swapped in each loop iteration. To hide this properly we have to hide both the random index `r` and which elements of `p` are modified by the swap.
# 
# We do so by representing `r` in a unary fashion, that is, as a unit vector `x_r` of length $n-i$. A unit vector is a list containing exactly one 1 and all other entries equal to 0. Taking the dot product between a secret unit vector `x_r` and a segment of `p` of the same length will select the intended element of `p`. At the same time no information whatsoever is given away about which element is selected.

# In[6]:


def random_permutation(n):                     # returns list of n secint elements
    p = [secint(i) for i in range(n)]          # initialize p to identity permutation
    for i in range(n-1):
        x_r = random_unit_vector(n-i)          # x_r = [0]*(r-i) + [1] + [0]*(n-1-r), i <= r < n
        p_r = mpc.in_prod(p[i:], x_r)          # p_r = p[r]
        d_r = mpc.scalar_mul(p[i] - p_r, x_r)  # d_r = [0]*(r-i) + [p[i] - p[r]] + [0]*(n-1-r)
        p[i] = p_r                             # p[i] = p[r]
        for j in range(n-i):
            p[i+j] += d_r[j]                   # p[r] = p[r} + p[i] - p[r] = p[i]
    return p                                


# Function `random_permutation` is defined as a plain Python function because we do not need to wait for any results explicitly. Two further functions from the `mpc` runtime are used:
# 
# 1. `mpc.in_prod(x, y)` to securely compute the dot product of `x` and `y`.
# 
# 2. `mpc.scalar_mul(a, x)` to securely compute the product of `a` with each element of `x`.
# 
# ## Random unit vectors
# 
# The final step is to generate uniformly random unit vectors of a given length. A unit vector of length $n$ is a bit vector with exactly one entry set to 1 and the remaining $n-1$ entries set to 0. 
# 
# Our algorithm for generating secret random unit vectors will be recursive. The basic idea is explained by a small example. Suppose we have generated $x = [0,0,1,0]$ as a secret unit vector of length $4$ (which should happen with probability 25%). Given $x$, we will output either $[0,0,0,0,0,0,1,0]$ or $[0,0,1,0,0,0,0,0]$ with 50% probability each, depending on a secret random bit. This way we are able to double the length at the expense of one secret random bit.
# 
# Let $n\geq2$, and suppose $n$ is even. We generate a secret random bit $b$ and, recursively, we generate a random unit vector $x$ of length $n/2$. We multiply $b$ with all elements of $x$, yielding vector $y$. So, for the example above, we get $y = [0,0,0,0]$ if $b=0$ and $y = x = [0,0,1,0]$ if $b=1$. Also, we see that $x-y = x$ if $b=0$ and $x-y = [0,0,0,0]$ if $b=1$. The correct output is thus obtained by taking the concatenation of $y$ and $x-y$.
# 
# For odd $n$, we try the same approach, but a slight problem arises. We cannot generate a vector of nonintegral length $n/2$. Instead, we recursively generate a random unit vector $x$ of length $m=(n+1)/2$. As before, we also generate a random bit $b$. Altogether, we thus get $2 m = n+1$ equally likely values for $b$ and $x$ jointly. Since our target is a uniformly random unit vector of length $n$, we need to reject one out of the $2m$ possible values for $b$ and $x$. Below we choose to reject in case $b=1$ and $x[0]=1$.
# 
# This leaves us with the case $n=1$, which is handled by simply returning $[1]$, the only unit vector of length $1$.

# In[7]:


@mpc.coroutine                                      # turn coroutine into an MPyC coroutine
async def random_unit_vector(n):                    # returns list of n secint elements
    await mpc.returnType(secint, n)                 # set return type of this MPyC coroutine
    if n == 1: 
        return [secint(1)]
    b = mpc.random_bit(secint)                      # pick one random bit if n>=2
    x = random_unit_vector((n + 1) // 2)            # recursive call with m=(n+1)//2
    if n % 2 == 0:
        y = mpc.scalar_mul(b, x)                    # y = [0]*m or y = x
        return y + mpc.vector_sub(x, y)             # [0]*m + x or x + [0]*m
    elif await mpc.eq_public(b * x[0], 1):          # reject if b=1 and x[0]=1
        return random_unit_vector(n)                # start over
    else:
        y = mpc.scalar_mul(b, x[1:])                # y = [0]*m or y = x[1:] 
        return x[:1] + y + mpc.vector_sub(x[1:], y) # [x[0]]  + ([0]*m + x[1:] or [0]*m + x[1:])


# Three more functions from the `mpc` runtime are used:
# 
# 1. `mpc.random_bit(sectype)` to generate a secret random bit of the given type.
# 
# 2. `mpc.vector_sub(x, y)` to securely compute the elementwise difference of `x` and `y`.
# 
# 3. `mpc.eq_public(a, b)` to securely test `a == b`, revealing only the outcome publicly.
# 
# Function `random_unit_vector` is also defined as an MPyC coroutine because we need to wait for the result of a coroutine call for a condition in an `if` statement. 
# 
# ## Test drive
# 
# Let's now check what the results look like. We check the first few cases for each function.

# In[8]:


N = 7


# In[9]:


print('Random unit vectors:')
for n in range(1, N + 1):
    s = mpc.run(mpc.output(random_unit_vector(n)))
    print(f'{n:2} {s}')


# In[10]:


print('Random permutations:')
for n in range(1, N + 1):
    s = mpc.run(mpc.output(random_permutation(n)))
    print(f'{n:2} {s}')


# In[11]:


print('Random derangements:')
for n in range(2, N + 1):
    s = mpc.run(mpc.output(random_derangement(n)))
    print(f'{n:2} {s}')


# In[12]:


mpc.run(mpc.shutdown())   # required only when run with multiple parties  


# In[ ]:


import sys; sys.exit()    # stop execution here when this notebook is run as a Python script, see below


# ## Deploying the Python code with multiple parties
# 
# All the Python code contained in this notebook can be saved as a Python script using the `Download as` option from the `File` menu in Jupyter notebooks. Choose to download the notebook as `Python (.py)`.
# 
# We have done so and the resulting file [SecretSantaExplained.py](SecretSantaExplained.py) is stored in the same directory as the present notebook. Now we can run the Python script, as shown below. 
# 
# First we show a run with one party only. However, this time the code runs outside the Jupyter notebook, using its own Python interpreter. Once the run is completed, the screen output is displayed in the output area below the cell:
# 
# 

# In[13]:


get_ipython().system('python SecretSantaExplained.py')


# Any MPyC program comes with several built-in command line options. So, let's take a look at the help message for [SecretSantaExplained.py](SecretSantaExplained.py).

# In[14]:


get_ipython().system('python SecretSantaExplained.py -H')


# Now for the "real thing". Let's do a run with three parties, in which three processes will be launched communicating via local tcp-connections. Let's set a few options as well. We enable SSL for the occasion, and we enforce that tcp port numbers will be used starting from 11443 (arbitrary choice). We also set the default bit length to $l=10$, which is large enough for our examples anyway. The default length of $l=10$ is now used because we use secure integers of type `mpc.SecInt()`; for secure integers of type `mpc.SecInt(32)`, say, changing $l$ has no effect.

# In[15]:


get_ipython().system('python SecretSantaExplained.py -M3 --ssl -B 11443 -L10')


# The code can be run with any number of parties $m=1,2,3,4,5,6,7,...$ of which no more than $t$ are assumed to be corrupt, with $0\leq t\leq \lfloor (m-1)/2 \rfloor$. As a final example, we show a run with $m=8$ parties, with the treshold set to $t=2$.

# In[16]:


get_ipython().system('python SecretSantaExplained.py -M8 -T2')


# This concludes the explanation of our MPyC protocol for the Secret Santa problem. The Python script [secretsanta.py](secretsanta.py) contains a slightly more extensive demo, showing how secure fixed-point arithmetic or secure finite fields can be used instead of secure integers.
