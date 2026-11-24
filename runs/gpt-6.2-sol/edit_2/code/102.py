import os
print(os.listdir('.'))
print(edited())
print([k for k in globals() if not k.startswith('_')])