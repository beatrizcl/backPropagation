#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#dataset = pd.read_csv('mnist_test.csv')


# In[12]:


import numpy as np
import pandas as pd

qtInter,qtOut,qtInput,e,alpha,epoca = 100,10,784,0,0.1,0
dataset = pd.read_csv('mnist_train.csv')
v = np.random.randn(qtInput, qtInter) / np.sqrt(qtInput)
w = np.random.randn(qtInter, qtOut) / np.sqrt(qtInter)
ativacao = 'sig'


# In[13]:


grafico = np.array([[0]*2])


# In[14]:


def trataLinha(inp):
    retorno = np.array(inp)
    return retorno/255

def targetVetor(labels, num_classes=10):
    retTarget = np.eye(num_classes)[labels]
    return retTarget

def inputZ(inp):
    inX = np.zeros(qtInter)
    for j in range (qtInter):
        inX[j] = np.sum(inp * np.array(v[:,j]))
    return inX

def inputY(inp,pesos):
    inX = np.zeros(qtOut)
    for k in range (qtOut):
        inX[k] = np.sum(inp * np.array(pesos[:,k]))
    return inX

def deltainJ (dK,peso):
    inJ = np.zeros(qtInter)
    for j in range (qtInter):
        for k in range (qtOut):
            inJ[j] += dK[k] * peso[j][k]
    return inJ

def funcAtivacao(x,func) :
    if (func == 'relu') : 
        return np.maximum(x,0) 
    elif (func == 'sig') : 
        return (1/(1+np.exp(-x)))
    elif (func == 'tanh'):
        return np.tanh(x)

def derivada(x,func):        
    if (func == 'relu') : 
        return (0 if x<0 else 1)
    elif (func == 'sig') : 
        return (x * (1 - x))
    elif (func == 'tanh') :
        return (1 - np.power(x,2))

def verificaAcerto(tar,out):
    indTar = np.unravel_index(np.argmax(tar, axis=None), tar.shape)
    indOut = np.unravel_index(np.argmax(out, axis=None), out.shape)
    if (indTar == indOut):
        return True
    else :
        return False

def deltaK(targetK,Yk,YinK,func) :
    dK = np.zeros(qtOut)
    for i in range (qtOut) :
        erro = targetK[i] - Yk[i]
        deriv = derivada(YinK[i],func)
        dK[i] = erro * deriv
    return dK

def deltaJ(inJ,inZ,func):
    delJ = np.zeros(qtInter)
    for j in range (qtInter):
        deriv = derivada(inZ[j],func)
        delJ[j] = inJ[j] * deriv
    return delJ

def deltaW(dK,Ze):
    dW = np.empty_like(w)
    for j in range(qtInter):
        for k in range(qtOut):
            dW[j][k] = alpha * dK[k] * Ze[j]
    return dW

def deltaV(linha,dJ):
    dV = np.empty_like(v)
    for i in range(qtInput):
        for j in range (qtInter):
            dV[i][j] = alpha * dJ[j] * linha[i]
    return dV


# In[15]:


def algoritmo(isTreino,qtEpoca):
    global epoca
    for i in range (qtEpoca):
        readLine(dataset,isTreino)
        epoca+=1


# In[16]:


def readLine(dSet,isTreino):
    global e
    r,e = 1,0
    for row in dSet.itertuples(index=False):
        linha = list(row)
        target = targetVetor(linha.pop(0))
        linha = trataLinha(linha)
        forward(linha,isTreino,target,r)
        r+=1
        
def forward(linha,isTreino,target,r):
    global e
    global grafico
    inZ = inputZ(linha)
    Z = funcAtivacao(inZ,ativacao)
    inY = inputY(Z,w)
    Y = funcAtivacao(inY,ativacao)
    if (not verificaAcerto(target,Y)):
        e+=1
    if isTreino:
        backPropagation(inZ,Z,inY,Y,target,linha)
    a = 100 - ((e/r)*100)
    if (r!=0 and r%20 == 0):
        print (target)
        print (Y)
        print (f'Linha: {r} Epoca: {epoca} Erros: {e}  Acerto: {a} %' + '\n' + '_________' )
        grafico = np.append(grafico,[[r,a]],axis=0)
    
def backPropagation(inZ,Z,inY,Y,target,linha):
    delK,inJ,delJ =[],[],[]
    delK = deltaK(target,Y,Y,ativacao) 
    inJ  = deltainJ(delK,w)
    delJ = deltaJ(inJ,Z,ativacao)
    atualizaPesos(delK,Z,linha,delJ)


# In[17]:


def atualizaPesos(delK,Z,linha,delJ):
    global w
    global v
    correcaoW = deltaW(delK,Z)
    correcaoV = deltaV(linha,delJ)
    w = corrigePeso(w,correcaoW)
    v = corrigePeso(v,correcaoV)
    
def corrigePeso(peso,delta):
    return np.add(peso,delta)


# In[ ]:


algoritmo(True,1)

