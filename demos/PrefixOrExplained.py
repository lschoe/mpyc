#!/usr/bin/env python
# coding: utf-8

# # Ten Ways to Do Prefix-Or: `mpyc.mpctools.accumulate()` Explained

# In[1]:


from mpyc.runtime import mpc
mpc.run(mpc.start()) # required only when run with multiple parties 


# This notebook explains the ideas behind the implementation of `accumulate(x, f)` used in MPyC, see [mpyc.mpctools.accumulate()](https://github.com/lschoe/mpyc/blob/785494285b35c84f2924e981685e526bba8e1e5b/mpyc/mpctools.py#L45-L89).
# 
# The critical performance measure for the use of `accumulate(x, f)` in multiparty computation is the round complexity (or, "depth").
# A secondary performance measure is the computational complexity (or, "work"), where we count the total number of applications of `f`.
# 
# We discuss the following range of solutions, where $n$ denotes the length of input list `x`:
# 
# 1. [Introduction](#1.-Introduction): problem specification and trivial solutions with linear depth and work of $n-1$
# 2. [A la Sklansky](#2.-A-la-Sklansky): minimal depth of $\log_2 n$, but super-linear $O(n \log n)$ work
# 3. [A la Brent&ndash;Kung](#3.-A-la-Brent-and-Kung): linear work of $\approx 2n$, with only slightly larger depth of $\approx 2\log_2 n$
# 4. [Conclusion](#4.-Conclusion): general solution for associative `f`, which cannot be both of $O(1)$ depth and linear (polynomial) work
# 
# In Sections 1&ndash;3 we focus on the special case of computing prefix-or, where `f = lambda a,b: a|b` is the logical or. In Section 4 we consider the general case of an arbitrary associative function `f`, which rules out well-known $O(1)$ depth solutions specific to prefix-or.

# ## 1. Introduction

# We start with a simple example of using function `accumulate()`.

# In[2]:


import mpyc.mpctools
list(mpyc.mpctools.accumulate(range(1, 11)))


# The numbers $1,3,6,\ldots,55$ are the partial (prefix) sums of the sequence $1,2,3,\ldots,10$.

# However, the same result can be obtained using the Python standard library:

# In[3]:


import itertools
list(itertools.accumulate(range(1, 11)))


# What is the point of reimplementing the function `accumulate()` in MPyC?
# 
# Well, if we would only use `accumulate()` to *add* numbers, there would not be much use for it. But if we use `accumulate()` to *multiply* numbers, as part of a multiparty computation, there will be a significant difference related to the **round complexity** of the implementation.
# 
# The same remark applies if we use `accumulate()` to compute the prefix-or for a list of bits, like this:

# In[4]:


import operator

def prefix_or0(x):
    return list(itertools.accumulate(x, operator.or_))

def prefix_or1(x):
    return list(mpyc.mpctools.accumulate(x, operator.or_))


# Given a list of bits `x`, both `prefix_or0(x)` and `prefix_or1(x)` compute the or over all (nonempty) prefixes of `x`.

# In[5]:


x = [0, 0, 0, 1, 0, 0, 1, 0]
print(prefix_or0(x))
print(prefix_or1(x))


# Once we reach the first (leftmost) `1` in the input, this bit and all succeeding bits will be `1` in the output.

# To see the difference between `prefix_or0(x)` and `prefix_or1(x)` when used in multiparty computation, we introduce a slightly modified version of MPyC's secure integer type. This way we count the total number of interactive evaluations of the operator `|` (same as `operator.or_`), which we will refer to as the **or-complexity**. Moreover, we keep track of the depth of each secure integer value computed along the way, which we will refer to as its **or-depth**. By definition, the input values are at depth 0. Each (intermediate) value resulting from an evaluation of `|` is at depth one more than the largest depth of its inputs, provided both inputs are secure integers.

# In[6]:


secint = mpc.SecInt(8)

class secint(secint):
    
    __slots__ = 'or_depth'
    
    or_complexity = 0

    def __init__(self, value=None):
        self.or_depth = 0
        super().__init__(value)
        
    def __or__(self, b):
        c = super().__or__(b)
        # self is a secure (secret-shared) integer
        if isinstance(b, secint):
            # b is a secure (secret-shared) integer as well
            c.or_depth = max(self.or_depth, b.or_depth) + 1  # one round of communication for secure or
            secint.or_complexity += 1
        else:
            # b is a public value
            c.or_depth = self.or_depth
        return c

    __ror__ = __or__


# To check the correctness and complexity of our prefix-or implementations we introduce two helper functions. Function `correctness(pf)` tests a given prefix-or function on a range of inputs. Function `complexity(pf, n)` determines the or-complexity and or-depth of `pf` for input lists of length `n`. 

# In[7]:


import random

def correctness(pf):
    l = secint.bit_length
    if l <= 8:
        # loop over all l-bit values in two's complement
        r = range(-2**(l-1), 2**(l-1))
    else:
        # loop over 100 random l-bit values in two's complement
        r = (random.randrange(-2**(l-1), 2**(l-1)) for _ in range(100))
    for i in r:
        i = mpc.input(secint(i), senders=0)
        x = mpc.to_bits(i)
        y = mpc.run(mpc.output(pf(x)))
        x = mpc.run(mpc.output(x))
        assert y == prefix_or0(x), (x, y, prefix_or0(x))
    print(f'{pf.__name__} OK')
    
def complexity(pf, n):
    # We take the all-zero list as input (complexity is data-independent anyway):
    x = [secint(0) for _ in range(n)]
    secint.or_complexity = 0
    y = pf(x)
    or_depths = [a.or_depth for a in y]
    print(f'or-complexity: {secint.or_complexity}')
    print(f'or-depth: {max(or_depths)} (per output element: {or_depths})')


# In[8]:


correctness(prefix_or0)
correctness(prefix_or1)


# In[9]:


complexity(prefix_or0, 22)


# This tells us that the or-complexity of `prefix_or0(x)` is linear in the length of `x`, which is perfectly fine.
# But it also shows that the or-depth is linear as well.
# 
# To understand why the or-depth of `prefix_or0()` is linear, we take a look at the following equivalent implementation `prefix_or2(x)`, which explicitly shows the dependencies between all applications of `|`. The first output element `y[0]` is just a copy of the first input element `x[0]`. The next output element is then computed as the or of the previous output element and the next input element:

# In[10]:


def prefix_or2(x):
    n = len(x)
    y = [None] * n
    y[0] = x[0]
    for i in range(1, n):
        y[i] = y[i-1] | x[i]
    return y


# In[11]:


correctness(prefix_or2)
complexity(prefix_or2, 8)


# We need 7 applications of `|` in total. These applications have to be evaluated in a strictly sequential order, which results in the increasing or-depths for the output elements. In general, for an input list `x` of length $n\geq1$, both the or-complexity and or-depth of `prefix_or2(x)` are equal to $n-1$, hence linear in $n$.

# Linear or-complexity is unavoidable, but linear or-depth (round complexity) is bad news. We want to have *sub-linear* round complexity, for instance, proportional to $\sqrt n$, or rather **logarithmic** round complexity proportional to $\log n$: otherwise, the wait time for exchanging secret-shares between the parties will probably dominate the overall performance.

# The logarithmic round complexity of `mpyc.mpctools.accumulate()` (as used in `prefix_or1()`) is what sets it apart from the linear round complexity of `itertools.accumulate()` (as used in `prefix_or0()`). We get the following results for $n=1,2,4,8,12,16$:

# In[12]:


complexity(prefix_or1, 1)
complexity(prefix_or1, 2)
complexity(prefix_or1, 4)
complexity(prefix_or1, 8)
complexity(prefix_or1, 12)  # not a power of 2
complexity(prefix_or1, 16)


# In general, the or-depth for $n=2^k$ with $k\geq0$ is equal to $k=\log_2 n$, which is favorable for many applications. A potential drawback is that the or-complexity is *super-linear* in $n$, equal to $(n/2)k = (n/2)\log_2 n$ for $n=2^k$. We explain the underlying method due to Sklansky in the next section.
# 
# To avoid the super-linear or-complexity for Sklansky's method, the implementation of `mpyc.mpctools.accumulate()` provides an alternative method, which is automatically selected for larger values of $n$, namely for $n\geq32$. Technically, there is also the requirement that PRSS (Pseudo-Random Secret Sharing) is disabled in MPyC.

# In[13]:


no_prss = mpc.options.no_prss
mpc.options.no_prss = True  # disable PRSS temporarily
complexity(prefix_or1, 32)
mpc.options.no_prss = no_prss


# The or-complexity is now reduced to $2n-2-\log_2 n$ for inputs of length $n=2^k$. The trade-off is that the or-depth increases by almost a factor of 2, as it becomes $2\log_2 n -2$. The underlying method due to Brent&ndash;Kung is explained in Section 3.

# ## 2. A la Sklansky

# J. Sklansky, *Conditional-Sum Addition Logic*, IRE Transactions on Electronic Computers, 9(6):226–231, 1960.

# The method due to Sklansky is actually pretty straightforward. We first split the input list into two halves and apply the function recursively to both halves. Then we update the second half of the output by or-ing each element with the last element of the first half.

# In[14]:


def prefix_or3(x):
    n = len(x)
    if n == 1:
        return x[:]
    
    y0 = prefix_or3(x[:n//2])
    y1 = prefix_or3(x[n//2:])
    a = y0[-1]
    return y0 + [a | b for b in y1]  # all |s in parallel in 1 round


# In[15]:


correctness(prefix_or3)
complexity(prefix_or3, 8)


# The actual implementation in `mpyc.mpctools.accumulate()` is slightly more advanced, as we want to avoid excessive copying of lists.

# In[16]:


def prefix_or4(x):
    def pf(i, j):
        h = (i + j)//2
        if i < h:
            pf(i, h)
            a = x[h-1]
            pf(h, j)
            x[h:j] = (a | b for b in x[h:j])  # all |s in parallel in 1 round
            
    n = len(x)
    x = x[:]
    pf(0, n)
    return x


# We analyze the complexity of `prefix_or4()` for input lists of length $n=2^k$, $k\geq0$, as follows.
# Let $T_n$ and $R_n$ denote the or-complexity and or-depth of `prefix_or4()`, respectively. For the or-complexity we have as recurrence relation $T_1 = 0$, $T_n = 2 T_{n/2} + n/2$ with solution $T_n = (n/2) \log_2 n$. And for the or-depth we have $R_1=0$ and $R_n=R_{n/2} +1$ with solution $R_n=\log_2 n$, as we need only one round at each level of the recursion.
# 
# For $n=8$ we thus get $(8/2) \log_2 8 = 12$ as or-complexity and $\log_2 8 =3$ as or-depth:

# In[17]:


correctness(prefix_or4)
complexity(prefix_or4, 8)


# ## 3. A la Brent and Kung

# R.P. Brent and H.T. Kung, *A Regular Layout for Parallel Adders*, IEEE Transactions on Computers, 31(3):260-264, 1982.

# To make the or-complexity linear in $n$ we introduce some auxiliary input and output parameters, allowing for reuse of computed values. We do so while retaining the logarithmic round complexity, essentially matching the result due to Brent&ndash;Kung.

# Our first step is to introduce a recursive function `pf(a, x)` with auxiliary input `a`. At each level of the recursion, parameter `x` will represent a segment of the original input. Parameter `a` should then represent the or of all values preceding that segment in the original input. We use the output for the left half as input for the second half:

# In[18]:


def prefix_or5(x):
    def pf(a, x):
        n = len(x)
        if n == 1:
            return [a | x[0]]
        
        y0 = pf(a, x[:n//2])
        y1 = pf(y0[-1], x[n//2:])
        return y0 + y1

    return pf(0, x)


# But this simple idea fails "miserably". Its performance is just as bad as for `prefix_or0()` above:

# In[19]:


correctness(prefix_or5)
complexity(prefix_or5, 22)


# The problem is that all evaluations of `|` are done sequentially in `prefix_or5()`, because `y0[-1]` is available only once the prefix-or for the first half has been completed.
# 
# To remove this dependency we introduce auxiliary output `b`. For `y, b = pf(a, x)`, parameter `b` will be the or of all bits in `x`, hence independent of `a`.

# In[20]:


def prefix_or6(x):
    def pf(a, x):
        n = len(x)
        if n == 1:
            return [a | x[0]], x[0]
    
        y0, b0 = pf(a, x[:n//2])
        y1, b1 = pf(a | b0, x[n//2:])
        return y0 + y1, b0 | b1

    return pf(0, x)[0]


# The auxiliary input for the second half becomes `a | b0`, to include the or `b0` over the first half. 

# In[21]:


correctness(prefix_or6)
complexity(prefix_or6, 1)
complexity(prefix_or6, 2)
complexity(prefix_or6, 4)
complexity(prefix_or6, 8)
complexity(prefix_or6, 16)


# We first determine the or-complexity $T'_n$ of `pf(a, x)` for lists `x` of length $n=2^k$ in case `a` is a `secint` value, for which we have $T'_1 = 1$ ,$T'_n = 2T'_{n/2} + 2$ with solution $T'_n = 3n -2$.
# 
# To determine the or-complexity $T_n$ of `prefix_or6(x)`, we see that `pf(a, x)`is called with public value `a=0`. Evaluations of `a | .` with `a=0` are for free, so we get $T_1 = 0$, $T_n = T_{n/2} + T'_{n/2} + 1 = T_{n/2} + 3(n/2) - 2 + 1 = T_{n/2} + 3n/2 - 1$ with solution $T_n=3n -\log_2 n - 3$.
# 
# Without proof we note that the or-depth is equal to $2 \log_2 n - 1$ for $n\geq2$. 

# The or-complexity $T_n$ of about $3n$ still includes some double work, however. For example, the output `pf(0, x)[1]` is simply discarded, as it is equal to `pf(0, x)[0][-1]`, hence redundant. To avoid these spurious computations, we define outputs `b, y = pf(a, x)` such that `y + [b]` is equal to the desired prefix-or. That is, we omit the last element from the output `y` compared to function `prefix_or6()` above.

# In[22]:


def prefix_or7(x):
    def pf(a, x):  # prefix or with a as initial value plus aux. output
        n = len(x)
        if n == 1:
            return [], x[0]

        y0, b0 = pf(a, x[:n//2])
        a_or_b0 = a | b0
        y1, b1 = pf(a_or_b0, x[n//2:])
        return y0 + [a_or_b0] + y1, b0 | b1
        
    y, b = pf(0, x)
    y.append(b)
    return y


# In[23]:


correctness(prefix_or7)
complexity(prefix_or7, 1)
complexity(prefix_or7, 2)
complexity(prefix_or7, 4)
complexity(prefix_or7, 8)
complexity(prefix_or7, 16)


# We first determine the or-complexity $T'_n$ of `pf(a, x)` for lists `x` of length $n=2^k$ in case `a` is a `secint` value, for which we have $T'_1 = 0$ ,$T'_n = 2T'_{n/2} + 2$ with solution $T'_n = 2n-2$.
# 
# To determine the or-complexity $T_n$ of `prefix_or6(x)`, we see that `pf(a, x)`is called with public value `a=0`. Evaluation of `a | .` with `a=0` is for free, so we get $T_1 = 0$, $T_n = T_{n/2} + T'_{n/2} + 1 = T_{n/2} + 2(n/2) - 2 + 1 = T_{n/2} + n - 1$ with solution $T_n=2n -\log_2 n - 2$.
# 
# The or-depth reduced to $R_n=2\log_2 n -2$ for $n\geq4$, with $R_1=0$ and $R_2=1$. To prove this we proceed as follows. 
# 
# First, we consider $R'_n$ defined as the or-depth of `pf(a, x)[1]` for a list `x` of length $n$. We have $R'_1=0$ and $R'_n=\max(R'_{n/2},R'_{n/2})+1=R'_{n/2}+1$ with solution $R'_n=\log_2 n$.
# 
# Next, we consider $R_n^{(d)}$ defined for $d\geq\log_2 n$ as the or-depth of `pf(a, x)[0]` including the or-depth of `a` given by $d$. We have $R_1^{(d)}=d$ and $R_n^{(d)}=R_{n/2}^{(\max(d, R'_{n/2})+1)} = R_{n/2}^{(d+1)}$ with solution $R_n^{(d)} = d+\log_2 n$.
# 
# Finally, we consider $R_n$ defined as the or-depth of `pf(0, x)`. We have $R_1=0$ and $R_n=\max(R_{n/2}^{(R'_{n/2})},R'_n)=\max(R_{n/2}^{(\log_2 (n/2))},\log_2 n)=\max(2 \log_2 (n/2), \log_2 n)$, which completes the proof.

# Our last goal is to eliminate list parameter `x` from the recursive calls, like we did for `prefix_or4()` above. The recursion in `prefix_or7()` uses at most two secure evaluations of `|`, which we want to preserve in the recursion below. The case `a=0` in `prefix_or7()` corresponds to the case `i=0` in the program below, so in this case the evaluation of `|` with `x[i-1]` can be skipped.

# In[24]:


def prefix_or8(x):
    def pf(i, j):
        h = (i + j)//2
        if i < h:
            pf(i, h)
            if i:
                x[h-1] |= x[i-1]
            pf(h, j)
            x[j-1] |= x[h-1]
        
    n = len(x)
    x = x[:]
    pf(0, n)
    return x


# In[25]:


correctness(prefix_or8)
complexity(prefix_or8, 1)
complexity(prefix_or8, 2)
complexity(prefix_or8, 4)
complexity(prefix_or8, 8)
complexity(prefix_or8, 16)


# We see that the or-complexity of `prefix_or8()` is fine. But the or-depth got much worse, in fact equal to the or-complexity, which means that all `|`s are done sequentially. Compared to  `prefix_or7()` we have apparently introduced an extra dependency. To track the problem we focus on `x[h-1]`: the recursive call `pf(i, h)` first sets `x[h-1]` to hold the or of segment `x[i:h]`, subsequently the or of the preceding part is included, and this value is then used to update `x[j-1]` at the end.
# 
# We fix the problem by noting that, as in `prefix_or7()`, we can use the value of `x[h-1]` as already set by the recursive call `pf(i, h)` to update `x[j-1]` at the end, that is, the update of `x[h-1]` is ignored:

# In[26]:


def prefix_or9(x):
    def pf(i, j):
        h = (i + j)//2
        if i < h:
            pf(i, h)
            b = x[h-1]
            if i:
                x[h-1] = x[i-1] | b
            pf(h, j)
            x[j-1] = b | x[j-1]
    
    n = len(x)
    x = x[:]
    pf(0, n)
    return x


# In[27]:


correctness(prefix_or9)
complexity(prefix_or9, 1)
complexity(prefix_or9, 2)
complexity(prefix_or9, 4)
complexity(prefix_or9, 8)
complexity(prefix_or9, 16)


# The implementation of `mpyc.mpctools.accumulate()` uses this approach.

# ## 4. Conclusion

# A.K. Chandra, S. Fortune, R. Lipton, *Unbounded Fan-in Circuits and Associative Functions*, 15th ACM Symposium on Theory of Computing (STOC '83), pp. 52–60, 1983.

# It is well-known that the optimal round complexity for prefix-or is $O(1)$, independent of the input length $n$. But these solutions do not generalize to arbitrary associative functions `f`, which is required for `accumulate(x, f)`. In fact, as shown by Chandra, Fortune, and Lipton (Theorem 2.1), one cannot have constant depth and polynomially bounded work.
# 
# Therefore, we opt for the above two methods of logarithmic depth. We choose Sklansky for smaller inputs (when the super-linear work is not too dominant). We also choose Sklansky when PRSS is enabled, giving depth minimization the highest priority. In all other cases we choose Brent&ndash;Kung, to keep the work linear at a modest increase in round complexity. See [mpyc.mpctools.accumulate()](https://github.com/lschoe/mpyc/blob/785494285b35c84f2924e981685e526bba8e1e5b/mpyc/mpctools.py#L45-L89) for the details.

# In[28]:


mpc.run(mpc.shutdown())  # required only when run with multiple parties 


# See also the executable Python script [PrefixOrExplained.py](PrefixOrExplained.py), which can be used to run all code contained in this notebook with multiple parties. For example, using `python PrefixOrExplained.py -M3`.
