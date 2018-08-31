#!/usr/bin/python
import matplotlib.pyplot as plt

def plot(files):
	f = open(files)
	lines = f.readlines()
	times = []
	for line in lines:
		times.append(float(line))

	plt.xlabel('log(o_channels)')
	plt.ylabel('time')
	plt.title("partD time--o_channel graphs")
	plt.plot([i for i in xrange(5)], times, marker = 'x', color = 'red', label=files)
	plt.legend()
	plt.savefig('partD_'+files[:-3]+'jpg')
	plt.cla()
	
	
def main():
	plot('trees.txt')
	plot('mountain.txt')

if __name__ == '__main__':
	main()
	
	
	