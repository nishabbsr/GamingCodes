#!/usr/bin/env python
# coding: utf-8

# In[5]:


from __future__ import division
import numpy as np

Q = [[3, -1,4], [-1, 2, 5], [4,5,6]]
b = [1,0,1]

x = [1,1,0]

gdvector = []
dirctn_vector = []
alpha = []

dirctn = np.dot(Q, x) + b
grad_dirctn = -dirctn

i = 1
while True:
    print ("iteration:", i)
    
    alpha_k = np.dot(r_k, r_k) / np.dot(np.dot(d_k, Q), d_k)
    x = x + np.dot(alpha_k, d_k)
    r_k1 = r_k + np.dot(np.dot(alpha_k,Q), d_k)
    beta_k1 = np.dot(r_k1, r_k1) / np.dot(r_k, r_k)
    d_k1 = -r_k1 + np.dot(beta_k1, d_k)

    r_k = r_k1
    d_k = d_k1
    
    if np.isclose(r_k, np.zeros(r_k.shape)).all():
        print ("Breaking!")
        print ("final r_k:", r_k)
        print ("final x:", x)
        break

    print( "r_k:",r_k)
    print ("x:", x)
    print 
    i += 1


# In[42]:


import numpy as np

def f(x, A, b):
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x) 

def optimal_grad(A, b, x=None):
    if not x:
        x = np.ones(len(b))
    function_value = [f(x, A, b)]

    direction = b + np.dot(A, x)
    grad_direction = direction
    directon_i_norm = np.dot(direction.T, direction)
    i = 0
    while True:
        A_grad_direction = np.dot(A, grad_direction)
        alpha_i = directon_i_norm / np.dot(grad_direction.T, A_grad_direction)
        x = x + alpha_i * A_grad_direction
        function_value.append(f(x, A, b)) 
        direction -= alpha_i * A_grad_direction
        directon_update_norm = np.dot(direction.T, direction)
        i += 1
        print('Iteration:', i)
        print('error', direction, 'Updated distance_norm', directon_update_norm)
        if np.sqrt(directon_update_norm) < 1e-8:
            print('Last Iteration:', i)
            break
        beta = directon_update_norm / directon_i_norm
        grad_direction = beta * grad_direction + direction
        directon_i_norm = directon_update_norm
    return x, function_value


# In[46]:


import gmtime, strftime 
   n = 3
   P = np.random.randint(1,20,size=[n, n])
   A = np.dot(P.T, P)
   b = np.ones(n)
   print(A)
   t1_axis = time.time()
   x1, function_value = optimal_grad(A, b)
   t1_axis = time.time()
   print(x1[:3])
   print("Final : %f (s)" % (t1_axis - t1_axis)) 

   


# In[16]:


P = np.randint(size=[n, n])
print(P)


# In[40]:


import numpy as np
import time
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def f(x, A, b):
    return 0.5 * np.dot(x.T, np.dot(A, x)) - np.dot(b.T, x) 

def conjugate_grad(A, b, x=None):
    if not x:
        x = np.ones(len(b))
    f_vals = [f(x, A, b)]

    r = b - np.dot(A, x)
    p = r
    r_i_norm = np.dot(r.T, r)
    i = 0
    # for i in range(n_iters * 2):
    while True:
        Ap = np.dot(A, p)
        alpha_i = r_i_norm / np.dot(p.T, Ap)
        x = x + alpha_i * p

        f_vals.append(f(x, A, b)) 

        r -= alpha_i * Ap
        r_i_plus_1_norm = np.dot(r.T, r)
        i += 1
        print('Itr:', i, 'updated x' , x)
        if np.sqrt(r_i_plus_1_norm) < 1e-8:
            print('Itr:', i)
            break
        beta = r_i_plus_1_norm / r_i_norm
        p = beta * p + r
        r_i_norm = r_i_plus_1_norm
        print('length of x', len(x))
        print('Length', len(f_vals))
    return x, f_vals

  


# In[43]:


n = 10
   # P = np.random.randint(1,20,size=[n, n])
P = np.random.normal(size=[n, n])
A = np.dot(P.T, P)
b = np.ones(n)
print(A)
t1_tic = time.time()
x1, f_vals1 = conjugate_grad(A, b)
t1_toc = time.time()
print(x1[:n])
print("Conjugate method: %f (s)" % (t1_toc - t1_tic))


# In[44]:


LEFT_LIMIT = 0
RIGHT_LIMIT = 1000
plt.plot(np.arange(len(f_vals1))[LEFT_LIMIT:RIGHT_LIMIT],
         np.array(f_vals1)[LEFT_LIMIT:RIGHT_LIMIT])
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
plt.xlim(LEFT_LIMIT, len(f_vals1))
plt.xlabel("Iteration time")
plt.ylabel("func value")
plt.legend()

plt.show()


# In[63]:


LEFT_LIMIT = 15
RIGHT_LIMIT = 1000
plt.plot(np.arange(len(f_vals1))[LEFT_LIMIT:RIGHT_LIMIT],
         np.array(f_vals1)[LEFT_LIMIT:RIGHT_LIMIT],
         label="CG")
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
plt.xlim(LEFT_LIMIT, len(f_vals1))
plt.xlabel("Iteration time")
plt.ylabel("func value")
plt.legend()

plt.show()


# In[ ]:




