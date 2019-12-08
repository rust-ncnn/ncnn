import re
reg_map = dict()

f = open("file")
pattern = re.compile(r'v\d{1,2}|x\d{1,2}|w\d{1,2}|\%w\d{1,2}|\%x\d{1,2}')

import pdb
while True:
    line = f.readline()
    if not line:
        break
    res = re.search(pattern, line)
    if res is not None:
        x = res.span()
        reg = line[x[0]:x[1]]
        reg_map[reg] = 1

keys = [ v for v in sorted(reg_map.keys())] 
print(keys)

for (k,v) in reg_map.items():
    print(k)
