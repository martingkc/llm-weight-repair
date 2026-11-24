print(sorted(k for k in globals() if not k.startswith('_')))
print(edited())
import os
print(os.listdir())