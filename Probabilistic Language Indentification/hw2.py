import sys
import math
import string

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    with open (filename,encoding='utf-8') as f:
        content = f.read().upper()

        alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for char in alpha:
            X[char] = 0

        for key in content:
            if key in X:
                X[key] += 1

    return X

print("Q1")
items = shred("letter.txt")
for key in items:
    print(key , items[key])
print("\n")

print("Q2")
value_1 = items['A']
english = value_1 * math.log(get_parameter_vectors()[0][0])
spanish = value_1 * math.log(get_parameter_vectors()[1][0])
print("%.4f" %english)
print("%.4f" %spanish)
print("\n")
 
print("Q3")
sigma_eng = 0.0
sigma_spa = 0.0
for i in range(0,26):
    key, value = list(items.items())[i]
    sigma_eng += value * math.log(get_parameter_vectors()[0][i])
    sigma_spa += value * math.log(get_parameter_vectors()[1][i])

f_English = math.log(0.6) + sigma_eng
f_Spanish = math.log(0.4) + sigma_spa

print("%.4f" %f_English)
print("%.4f" %f_Spanish)
print("\n")  

print("Q4")
ratio = 0.0
if f_Spanish - f_English >= 100: 
    ratio = 0
elif f_Spanish - f_English <= -100: 
    ratio = 1
else:
    ratio = 1/(1 + math.exp(f_Spanish - f_English))

print("%.4f" %ratio)
 

