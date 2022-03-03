file = open('3240.log','r')
val = []
test = []
test_line = 0
test_metirc = []
metric = []
for line in file:
    parts = line.strip().split(' ')
    # average val loss
    if parts[0]=='average' and parts[1] == 'val':
        val.append(float(parts[3]))
    if test_line>0:
        test_line-=1
        metric.append(line)
    if parts[0]=='average' and parts[1] == 'test':
        if len(metric)>0:
            test_metirc.append(metric)
        metric = []
        test_line=3
        test.append(float(parts[3]))
    
import numpy as np
val = np.array(val)
print("min val",np.min(val))
print("test MSEL",test[np.argmin(val)])
print(test_metirc[np.argmin(val)])