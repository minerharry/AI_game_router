import random
import os

def f1(n):
    result = [];
    for k in range(n):
        for j in range(k):
            result.append(k**j);
    return result;

def f2(n):
    result = [];
    for k in range(n):
        for j in range(k):
            result.append(k**j);
    return result;

def f3(n):
    result = [];
    for k in range(n):
        for j in range(k):
            result.append(k**j);
    return result;

def f4(n):
    result = [];
    for k in range(n):
        for j in range(k):
            result.append(k**j);
    return result;

def f5(n):
    result = [];
    for k in range(n):
        for j in range(k):
            result.append(k**j);
    return result;
print(os.getpid());
while True:
    n = random.randrange(1,100);
    c = random.choice([f1,f2,f3,f4,f5]);
    result = c(n);