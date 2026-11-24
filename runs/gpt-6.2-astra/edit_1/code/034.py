import os
print(os.getcwd(),os.listdir('.')[:300]); print([k for k in globals() if not k.startswith('_')])