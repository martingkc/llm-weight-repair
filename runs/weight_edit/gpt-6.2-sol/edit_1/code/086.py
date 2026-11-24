for fn in [body_offset,logit_offset,exact_prefix,sleepy,make_write,block_ana,hiddenlens]:
 print(fn.__name__,inspect.signature(fn),fn.__code__.co_names,fn.__code__.co_consts[:65]);
print([len(qa) for qa in [PRACTICE['medical'],PRACTICE['general']]])