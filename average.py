from __future__ import division
import sys

num = int(sys.argv[1])
results = []

for i in range(num):
	fileLog = open('cuda_mm_' + str(i) + '.txt')
	fileLines = fileLog.readlines()
	results.append(float(fileLines[0].split(' ')[2]))
	
average = sum(results)/num

print 'Resultados dos testes:'
print results
print 'Media dos resultados:'
print average
