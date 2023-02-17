#!/usr/bin/env python
# coding: utf-8

# # Q1. 

#  입력으로 정수 N개로 이루어진 수열 A이 주어졌을 때 X이상인 수를 출력하는 프로그램을 구현하세요.

# In[1]:


X = int(input("X:"))
N = int(input("N:"))
# X, N 의 값을 입력받는다.


# In[2]:


import numpy as np
import random
#배열을 만들 때 필요한 패키지를 가져온다


# In[3]:


A = np.arange(N)
random.shuffle(A)
print('A:',A)
#1~N 순서의 배열을 만든 다음, 배열을 섞은 후, 프린트한다.


# In[4]:


B = [] 
#아무 값이 없는 B배열을 하나 만들어놓는다.
for i in range(0,N):
    if (A[i]>X):
        B.append(A[i])
#A의 원소값이 X보다 클 경우에만 B 배열에 추가되도록 if 문을 활용하였다.
for i in B:
    print(i, end=" ")
#end =" "을 넣음으로써, 기본값인 \n 을 빼, 좀더 보기 편하게 하였습니다.


# # Q2.

# 주어진 CSV 파일을 불러오고, pandas와 matplotlib 패키지를 사용해 파일의 x, y좌표를 2차원 좌표계에 선그래프로 시각화 하세요.

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt 
#dataframe과 함수를 그리는 데 필요한 패키지를 가져온다.


# In[6]:


A = pd.read_csv(r"C:\Users\Jungro\Desktop\data.csv")

plt.plot(A.iloc[:,0],A.iloc[:,1])
#slicing을 활용하여 그래프를 그려준다


# # Q3. 

# 다음 조건을 만족하는 함수를 define 하세요.

# - 함수의 이름은 $Cosine()$입니다.
# - 사용자는 삼각형의 변의 길이 $a,b,c$를 입력합니다.
# - 만일, 삼각형 $ABC$가 만들어지지 않는다면 NO를, 만들어 지는 경우에는 $Cos A$를 출력합니다.

# In[7]:


a = int(input("a:"))
b = int(input("b:"))
c = int(input("c:"))
A = [a,b,c]

# a, b, c 값을 받은 후 리스트에다가 넣는다.
def Cosine(a,b,c):
    if max(A)>(sum(A)-max(A)):
        print('NO')
    #삼각형의 조건인 '두 변의 길이의 합은 다른 한 변의 길이보다 커야한다'를 활용
    else:
        print('CosA의 값은:',(b**2 + c**2 - a**2)/(2*b*c));        
    #CosA의 수식 프린트    

Cosine(a,b,c)


# # Q4.

# 입력으로 한 점 A의 좌표와 사각형의 네 점의 좌표가 주어졌을 때 한 점에서 사각형까지의 최단 거리 L을 구하는 프로그램을 작성하세요.

# In[8]:


x1,x2,x3,x4,x5,y1,y2,y3,y4,y5 = map(float,input("각 좌표의 x, y 값은:").split(" "))


# In[9]:


A = [x1,y1]; C1 = [x2, y2]; C2 = [x3 , y3]; C3 = [x4, y4]; C4 = [x5, y5]


# In[10]:


plt.plot(x1,y1,'go')
plt.plot([ x2, x3, x4, x5], [ y2, y3, y4, y5],'o')
plt.show()

# A와 C1, C2, C3, C4의 위치를 그래프에 표시하였다. A만 여기서 초록색을 띄고 있다.


# In[11]:


import math


# 주어진 A 좌표와 C1, C2, C3, C4가 있다. 
# 그리고 그 사각형까지의 최소거리를 계산하기 
# 위해서 사각형 안의 모든 점들을 이은 선분까지의 거리를 계산한 후, 그 중 최소값을
# 소수점 둘째 자리까지 프린트하는 절차를 취할 것이다.

# In[12]:


def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(((x2 - x1)**2)+ ((y2 - y1)**2))
    return lineMagnitude

def DistancePointLine (px, py, x1, y1, x2, y2):
    
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    # 점에서 내린 수선의 발이 그 선분 위에 없을 때 제일 가까운 거리를 반환한다.        
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
     # 점에서 내린 수선의 발 까지의 거리 
    return DistancePointLine


# In[13]:


a = DistancePointLine(x1, y1, x2, y2, x3, y3)
b = DistancePointLine(x1, y1, x2, y2, x4, y4)
c = DistancePointLine(x1, y1, x2, y2, x5, y5)
d = DistancePointLine(x1, y1, x3, y3, x4, y4)
e = DistancePointLine(x1, y1, x3, y3, x5, y5)
f = DistancePointLine(x1, y1, x4, y4, x5, y5)
#A에서 각 선분까지의 최소거리


# In[14]:


D = [a, b, c, d, e, f]
D.sort()
print(round(D[0],2))
# 각 선분까지의 최소거리를 리스트 안에 넣은다음, 오름차순으로 정리를 하였다.


# # Q5. 

# 입력으로 정수 N가 대칭수(Palindrome Number)인지 판별하는 프로그램을 구현하세요. 
# - 정수 N를 입력받습니다.
# - 만약 N이 대칭수이면 True, 아니면 False를 출력합니다.

# In[17]:


N = str(input("N:"))
A = -int(N)
#입력은 문자열 형식으로 받는다.   


# In[18]:


Answer = 'True'
if int(N)<0:
    print('%s 을 대칭시키면 ' '%i' '-이므로 %s은 대칭수가 아니다.'%(N,A,N))
# 0보다 작은 정수에 대해서는 대칭이 안 되는 이유를 설명해준다.    
else:    
    for i in range(len(N)//2):
        if N[i] != N[-1-i]:
            Answer = 'False'
            break
    print(Answer)     

