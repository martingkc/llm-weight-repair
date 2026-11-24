import inspect
print(inspect.signature(rnorm)); print(rnorm.__code__.co_consts, rnorm.__code__.co_names); print(inspect.signature(tops)); print(tops.__code__.co_names, tops.__code__.co_consts)