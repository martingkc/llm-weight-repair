import os,inspect
print(os.listdir()); print([x for x in globals() if not x.startswith('_')]); print(CONFIG); print(edited())