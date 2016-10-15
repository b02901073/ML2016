# -*- coding: utf-8 -*- 
import csv
#import numpy as nd
traindata = []
with open('train.csv',newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        traindata.append(row)
f.close()
setdata=[]
hour=[]
for i in range(0,12): #12 month
	for j in range(0,20): #20 days
		for k in range(0,24):
			for l in range(0,18):
				if traindata[1+i*360+j*18+l][3+k]=="NR":
					traindata[1+i*360+j*18+l][3+k]=0
				hour.append(traindata[1+i*360+j*18+l][3+k])
			setdata.append(hour)
			hour=[]
print(setdata[500])

temp=[]
data=[]

for i in range(0,12):
	for j in range(0,471):
		for k in range(0,9):
			for l in range(0,18):
				temp.append(setdata[i*480+j+k][l])
		temp.append(setdata[i*480+j+9][9])
		data.append(temp)
		#print(setdata[i*480+j+9][9])
		temp=[]
#print(len(data[0]))
rate=0.0000005 #learning rate

b=0
b_learn=0
w =[] #各參數
learning=[]
for i in range(0,162):
	w.append(1)
flag=1
Loss=0
#print(len(w))
#計算Loss function
for a in range(0,50000):
	for i in range(0,5652):
		y=float(data[i][162])
		#print(y)
		temp=b
		for j in range(0,162):
			temp=temp+w[j]*float(data[i][j])
		Loss=Loss+(y-temp)*(y-temp)
		#print(temp)
		b_learn=2*(y-temp)*rate
		for k in range(0,162):
			x=2*(y-temp)*float(data[i][k])*rate
			#print(x)
			learning.append(x)
		b+=b_learn
		for l in range(0,162):
			w[l]=w[l]+learning[l]
		learning=[]
	print(a)
	print(b)
	print(Loss)
	Loss=0
		#print(w[j])
		#print(float(data[i][j]))
		#temp=temp+w[j]*data[i][j]

testdata = []
with open('test_X.csv',newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        testdata.append(row)
f.close()

submit = []
submit.append(["id","value"])
station = []
for i in range(0,240):
	for j in range(0,9):
		for k in range(0,18):
			if testdata[i*18+k][j+2]=="NR":
				testdata[i*18+k][j+2]=0
			station.append(testdata[i*18+k][j+2])
			#print(testdata[i*18+k][j+2])
	#print(len(station))
	y=b
	for l in range(0,162):
		y=y+w[l]*float(station[l])
	submit.append(["id_"+str(i),y])
	#print(station)
	station=[]

f = open("linear_regression.csv","w",newline='')
w = csv.writer(f)
w.writerows(submit)
f.close()

f = open("kaggle_best.csv","w",newline='')
w = csv.writer(f)
w.writerows(submit)
f.close()