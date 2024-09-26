# 可以容在optimize里面
import numpy as np
import pandas as pd
file_path = 'a_dict_data.xlsx'
df = pd.read_excel(file_path)
data=[]
temp=[None]*6
endtile=[]
start=[]
end=[]
for _, row in df.iterrows():
    tilenum = row['tile num']#16headbc
    data.append(tilenum)

for i in range(10):
    temp = data[i*6:(i+1)*6]
    #print(temp)
    endtile =  [sum(temp[:j+1]) for j in range(len(temp))]
    starttile = endtile[:]
    starttile =  [endtile[j]+1 for j in range(len(starttile))]
    starttile.pop()
    starttile.insert(0,1)
    start.extend(starttile)
    end.extend(endtile)
a=[0,1,2,3,4,5]   
start.extend(a)
end.extend(a)
df['start tile'] = start
df['end tile'] = end
df.to_excel(file_path, index=False)