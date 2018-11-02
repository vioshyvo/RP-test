import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import glob
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18



def main(k, files):
    depth = 10

    fig = plt.figure()
    LSD = []
    q = False
    A = []
    ax = plt.gca()

    for resfile in files:
        with open(resfile) as f:
            _ = f.readline()
            lines = [[float(x) for x in s.strip().split()] for s in f]
        index_time = [x[9] for x in lines if x[0] == k and x[2] == depth]
        n_trees = [x[1] for x in lines if x[0] == k and x[2] == depth]

        index_time2 = []
        n_trees2 = []
        included_trees = set()
        for i in range(len(n_trees)):
            if n_trees[i] not in included_trees:
                included_trees.add(n_trees[i])
                n_trees2.append(n_trees[i])
                index_time2.append(index_time[i])

        A.append((n_trees2, index_time2))

    for a in A:
        print(a)
        print("\n\n")

    colors = cm.rainbow(np.linspace(0, 1, len(A)))
    minY, maxY = float('inf'), -float('inf')
    for a, c, m in zip(A, colors, ['>', 'v', 'd', '^', 'o', 'p', 'h', '<']):
        print a, c, m
        l, = ax.plot(a[0], a[1], linestyle='solid', marker=m, label=a[0], c=c, markersize=7)

    ax.legend(LSD, labels=files, loc="upper left", title = 'file name')

    plt.show()



if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2:])
