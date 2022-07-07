import heapq
import numpy as np
from mpi4py import MPI 


def objectiveValue(tableau):
   if tableau is not None:
      return -(tableau[-1][-1])
   return 

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
   
   if(quotients):
       row = min(quotients, key=lambda x: x[1])[0] 
   
   return row, column

def tableau_gen(r, c):
   A = np.random.randint(1, 20, size = (r, c))
   slacks = np.eye(r, c-1, 0)
   b = np.random.randint(1, 50, size = r-1)
   b = np.append(b, 0).reshape(r, 1)
   tableau = np.hstack((A, slacks))
   tableau = np.hstack((tableau, b))
   return tableau

def pivotAbout(recvbuf, pivot, pivotRow, rank):

   if rank != pivot[0]//ROWSINPROC:
      for row_idx, row_ in enumerate(recvbuf):
         pivotRowMultiple = [y * recvbuf[row_idx][pivot[1]] for y in pivotRow]
         recvbuf[row_idx] = [x - y for x, y in zip(recvbuf[row_idx], pivotRowMultiple)]
   
def pivotRow(recvbuf, pivot, rank): 
   
   if rank == pivot[0]//ROWSINPROC:
      piv_row_idx = pivot[0]-rank*ROWSINPROC
      pivotDenom = recvbuf[piv_row_idx][pivot[1]]
      recvbuf[piv_row_idx] = [x / pivotDenom for x in recvbuf[piv_row_idx]]
      for row_idx, row_ in enumerate(recvbuf):
         if row_idx == piv_row_idx: 
            continue
         else:
            pivotRowMultiple = [y * recvbuf[row_idx][pivot[1]] for y in recvbuf[piv_row_idx]]
            recvbuf[row_idx] = [x - y for x, y in zip(recvbuf[row_idx], pivotRowMultiple)]
      
      return recvbuf[piv_row_idx]

def mpi_simplex(firstIter, recvbuf = None, pivot = None):   
   tableau_ = None
   pivot_ = None
   row = None
   canImprove_ = None
   recvbuf_ = np.empty((ROWSINPROC, COLUMNS*2))

   if rank == 0:
      if firstIter:
         tableau_ = tableau_gen(ROWS, COLUMNS)
         print('Given tableau:\n',tableau_,'\n')
      
      if pivot is None:
         pivot_ = findPivotIndex(tableau_)
      else: pivot_ = pivot

   if recvbuf is None:
      comm.Scatter(tableau_, recvbuf_, root=0)
   else: recvbuf_ = recvbuf

   if pivot is None:
      pivot_ = comm.bcast(pivot_, root=0)
   else: pivot_ = pivot

   row = pivotRow(recvbuf_, pivot_, rank)

   row = comm.bcast(row, pivot_[0]//ROWSINPROC)

   pivotAbout(recvbuf_, pivot_, row, rank)

   if rank == 0:
      tableau_ = np.empty((ROWS, COLUMNS*2))

   comm.Gather(recvbuf_, tableau_, root=0)

   if rank == 0:
      if canImprove(tableau_):
         canImprove_ = True
         pivot_ = findPivotIndex(tableau_)
         comm.Scatter(tableau_, recvbuf_, root=0)
      else: 
         print('Final tableau:\n', tableau_)
         print('Objective value:',round(objectiveValue(tableau_), 2))

   pivot_ = comm.bcast(pivot_, root=0)
   canImprove_ = comm.bcast(canImprove_, root=0)

   return canImprove_, recvbuf_, pivot_

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
PROCS = comm.Get_size()

ROWS, COLUMNS = 80, 80
ROWSINPROC = np.int32(ROWS/PROCS)

def mpi():
   canimp, recvbuf, pivot = mpi_simplex(True)
   if rank == 0:
      startwtime = MPI.Wtime()
      iterations = 1
   while canimp:
      canimp, recvbuf, pivot = mpi_simplex(False, recvbuf, pivot)
      if rank == 0:
         iterations += 1
   if rank == 0:
      endwtime = MPI.Wtime()
      totaltime = round((endwtime - startwtime), 3)
      print('Number of iterations:',iterations)
      print ('Time spent:',totaltime)
      timeperiter = totaltime*1000/iterations
      print('Time per iteration:',timeperiter)

mpi()

