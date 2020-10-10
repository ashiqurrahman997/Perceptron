import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(1234567)

def batch(wt, y, eta):
    count=1;
    sum,sum1=0,0;
    while True :
        g_y =np.dot(y,wt)
        index = np.array(np.where(g_y<=0))
        if (index.shape[1] != 0):
            count=count+1;
            sum=y[index[0,:],:].sum(axis=0)
            wt = wt+eta * (sum+sum1)
            sum1=sum+sum1;
        else :
            break;
    return count;

def single(wt,y,eta):
    count = 0;
    while True:
        p=0;
        for i in range(0,y.shape[0]):
            g_y = np.dot(wt,y[i,:])
            if (g_y <= 0):
                wt = wt + eta * y[i];
            else:
               p=p+1 ;
        count = count + 1;
        if(p==y.shape[0]):
            break;
    return count;

print("Ashiqur Rahman, ID : 150204057")

df_train = pd.read_csv('train.txt', sep=" ", header = None)
df_train = pd.DataFrame(df_train.values, columns = ['X', 'Y', 'Class'])
df_train=df_train.sort_values(by=['Class'])
print(df_train)
df=df_train.iloc[:,0:2].values

df1=df_train[df_train['Class'] == 1];
df1=df1.iloc[:,0:2]
w1=df1.values

df2=df_train[df_train['Class'] == 2];
df2=df2.iloc[:,0:2]
w2=df2.values

y= np.empty(shape=[0, 6])

for i in range(0,df_train.shape[0]):
    m = np.array([[df[i, 0] * df[i, 0], df[i, 1] * df[i, 1], df[i, 0] * df[i, 1], df[i, 0], df[i, 1], 1]]);
    y = np.append(y, m, axis=0)

y=np.concatenate((y[0:np.shape(w1)[0],:],-1*y[np.shape(w1)[0]:,:]),axis=0)


wt_0=np.array([0,0,0,0,0,0])
wt_rnd=np.ceil(10*np.array(np.random.rand(6)))
wt_1=np.array([1,1,1,1,1,1])

batch_result_0=[]
single_result_0=[]
batch_result_1=[]
single_result_1=[]
batch_result_rnd=[]
single_result_rnd=[]

alpha=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]);


for i in range(0,alpha.shape[0]):
    batch_result_0.append( batch(wt_0, y, alpha[i]) )
for i in range(0,alpha.shape[0]):
    single_result_0.append(single(wt_0,y,alpha[i]))

for i in range(0, alpha.shape[0]):
    batch_result_1.append(batch(wt_1, y, alpha[i]))

for i in range(0, alpha.shape[0]):
    single_result_1.append(single(wt_1, y, alpha[i]))

for i in range(0, alpha.shape[0]):
    batch_result_rnd.append(batch(wt_rnd, y, alpha[i]))
for i in range(0, alpha.shape[0]):
    single_result_rnd.append(single(wt_rnd, y, alpha[i]))

print('Task 2 : Y=')
print(y)

print('Task:3')
print("Case 1:Initial Weight Vector All One")
dataset1 = pd.DataFrame({'Value of Alpha(learning Rate) ': alpha, ' One at a time  ': single_result_1,'Many at a time':batch_result_1})
print(dataset1.to_string(index=False))

print("Case 2:Initial Weight Vector All Zero ")
dataset1 = pd.DataFrame({'Value of Alpha(learning Rate) ': alpha, ' One at a time  ': single_result_0,'Many at a time':batch_result_0})
print(dataset1.to_string(index=False))

print("Case 3:Initial Weight Vector All Random ")
dataset1 = pd.DataFrame({'Value of Alpha(learning Rate) ': alpha, ' One at a time  ': single_result_rnd,'Many at a time':batch_result_rnd})
print(dataset1.to_string(index=False))


plt.figure(0);
plt.scatter(w1[:,0],w1[:,1],color = 'red', marker = 'o',label="w1")
plt.scatter(w2[:,0],w2[:,1],color = 'blue', marker = '+',label="w2")

plt.title(" Perceptron Algorithm task-1")
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend(loc="best",fontsize="small")

plt.figure(1)
plt.bar(alpha,batch_result_0,0.03,align='edge',color='blue',label='Many At a times')
plt.bar(alpha,single_result_0,0.03,align='center',color='red',label='One At a times')
plt.title('All values are Zeros')
plt.xlabel("Learning Rate")
plt.ylabel("Iteration Number")
plt.legend(loc="best",fontsize="small")


plt.figure(2)
plt.bar(alpha,batch_result_1,0.03,align='edge',color='green',label='Many At a times')
plt.bar(alpha,single_result_1,0.03,align='center',color='yellow',label='One At a times')
plt.title('All values are Ones')
plt.xlabel("Learning Rate")
plt.ylabel("Iteration Number")
plt.legend(loc="best",fontsize="small")

plt.figure(3)
plt.bar(alpha,batch_result_rnd,0.03,align='edge' ,color='black',label='Many At a times')
plt.bar(alpha,single_result_rnd,0.03,align='center',color='pink',label='One At a times')
plt.title('Random Values')
plt.xlabel("Learning Rate")
plt.ylabel("Iteration Number")
plt.legend(loc="best",fontsize="small")

plt.legend()
plt.show()
