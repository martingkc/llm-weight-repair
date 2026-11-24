import os
print(os.getcwd(), os.listdir())
print([k for k in globals() if not k.startswith('_')])