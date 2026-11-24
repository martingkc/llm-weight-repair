import os, inspect
print([(k,type(v).__name__,getattr(v,'shape',None)) for k,v in globals().copy().items() if not k.startswith('_')][-330:]);print(os.listdir('.'))