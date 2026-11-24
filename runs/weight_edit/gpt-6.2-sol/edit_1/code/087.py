dis.dis(get_xset,depth=0)
print([(fn.__name__,inspect.signature(fn),fn.__code__.co_consts[:30]) for fn in [axis_sep,reconstruct_x,rampspec]])