import json
from itertools import chain




#initializing q values and creating a q-value json file
qval = {}
# X -> [-40,-30...120] U [140, 210 ... 490]
# Y -> [-300, -290 ... 160] U [180, 240 ... 420]
for x in chain(list(range(-40, 140, 10)), list(range(140, 421, 70))):
    for y in chain(list(range(-300, 180, 10)), list(range(180, 421, 60))):
        for v in range(-10, 11):
            qval[str(x) + "_" + str(y) + "_" + str(v)] = [0, 0]


fd = open("q_learning_values.json", "w")
json.dump(qval, fd)
fd.close()