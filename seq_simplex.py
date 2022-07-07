import heapq
import numpy as np
from timer import Timer

def objectiveValue(tableau):
   return -(tableau[-1][-1])

def canImprove(tableau):
   lastRow = tableau[-1]
   return any(x > 0 for x in lastRow[:-1])

def moreThanOneMin(L):
   if len(L) <= 1:
      return False

   x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
   return x == y

def findPivotIndex(tableau):

   column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
   column = min(column_choices, key=lambda a: a[1])[0]

   if all(row[column] <= 0 for row in tableau):
      raise Exception('Linear program is unbounded.')

   quotients = [(i, r[-1] / r[column])
      for i,r in enumerate(tableau[:-1]) if r[column] > 0]

   if moreThanOneMin(quotients):
      raise Exception('Linear program is degenerate.')

   row = min(quotients, key=lambda x: x[1])[0]

   return row, column

def pivotAbout(tableau, pivot):
   i,j = pivot

   pivotDenom = tableau[i][j]
   tableau[i] = [x / pivotDenom for x in tableau[i]]

   for k, row in enumerate(tableau):
      if k != i:
         pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
         tableau[k] = [x - y for x, y in zip(tableau[k], pivotRowMultiple)]

def tableau_gen(r, c):
   A = np.random.randint(1, 20, size = (r, c))
   slacks = np.eye(r, c-1, 0)
   b = np.random.randint(1, 50, size = r-1)
   b = np.append(b, 0).reshape(r, 1)
   tableau = np.hstack((A, slacks))
   tableau = np.hstack((tableau, b))
   return tableau

def seq_simplex(r, c):
   tableau = tableau_gen(r, c)
   print("Tableau size - {}x{}".format(r, c))
   print('Given tableau:\n',tableau)
   iterations = 0
   while canImprove(tableau):
      pivot = findPivotIndex(tableau)
      pivotAbout(tableau, pivot)
      iterations = iterations + 1
   print('Final tableau:\n', tableau)
   print('Objective value:', round(objectiveValue(tableau), 2))
   print('Number of iterations:', iterations)
   return iterations
   

timer = Timer()

def test(num):
   totalTime = 0
   totalIters = 0
   for i in range(num):
      timer.start()
      iter = seq_simplex(ROWS, COLUMNS)
      timer.stop()
      totalIters = totalIters + iter
      totalTime = totalTime + timer.elapsed
      i += 1
   print('Average time spent:', totalTime/num)
   print('Average iterations number:', totalIters/num)
   print('Average time per iteration:', (totalTime/totalIters)*1000)


ROWS = 120
COLUMNS = 120
seq_simplex(ROWS, COLUMNS)
test(10)