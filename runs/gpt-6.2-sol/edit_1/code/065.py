for fn in [make_write,add_slots,edit_routes,edit_guard,edit_polar,top_base_path,exact_prefix,hiddenlens,block_ana]:
 print(fn.__name__,inspect.signature(fn),fn.__code__.co_names,fn.__code__.co_consts[:40]);
print('meta',list(META.keys()),'adapted',type(ADAPTED),len(ADAPTED)); print('finetuning artifacts?',type(lora),type(delta));
pa19=json.load(open('pass_19.json'))
for row in pa19[:18]:print(row['question'],'>',row['answer'])