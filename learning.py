import cleaningtest1 as np
a=np.ones((5,5),dtype='int32')
a[1:4,1:4]=np.zeros((3,3),dtype='int32')
a[2,2]=9

print(a)