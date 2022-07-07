from scipy.optimize import linprog
c = [-13, -10, -16]
A = [[10, 15, 10], [5, 5, 14]]
b = [38, 37]
res = linprog(c, A_ub=A, b_ub=b,bounds=(0, None))
print("Objective value = {}".format(res.get('fun') * -1))  
