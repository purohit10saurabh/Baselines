import sys

root = open(sys.argv[1],'r')
file = open(sys.argv[2],'w')
inst = root.readline()
counter = int(inst.split(' ')[0])
for x in range(counter):
	line=root.readline().strip().split(' ')
	if len(line[0].split(":")) > 1 or line[0]=='':
		continue
	else:
		print(' '.join(line),file=file)

print(' '.join(inst.split(' ')[1:]))