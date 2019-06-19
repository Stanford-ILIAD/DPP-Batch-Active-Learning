import demos
import sys

task   = sys.argv[1].lower()
method = sys.argv[2].lower()
N = int(sys.argv[3])
M = int(sys.argv[4])

print('Running in the automated mode: Reward will be randomly generated. Check simulation_utils.py to change that.')
if method == 'nonbatch' or method == 'random':
    demos.nonbatch(task, method, N, M)
elif method == 'greedy' or method == 'medoids' or method == 'boundary_medoids' or method == 'successive_elimination' or method == 'dpp':
    b = int(sys.argv[5])
    demos.batch(task, method, N, M, b)
else:
    print('There is no method called ' + method)

