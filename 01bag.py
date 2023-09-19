N = 3
C = 4
weight = [1, 2, 3]
value = [12, 10, 12]

def best_value(w, v , i ,c):
    if i<0 or c<=0:
        return 0
    res=best_value(w,v,i-1,c)
    if c>=w[i]:
        res=max(res,best_value(w,v,i-1,c-w[i])+v[i])
    return res

res=best_value(weight,value,N-1,C)
print(res)