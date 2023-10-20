import os

os.chdir('./gauss_70_2500/lucy')

for i in range(0,2500):
	old = 'out_' + str(i) + '.txt'
	new = 'out_' + str(i + 2500) + '.txt'
	os.rename(old, new)