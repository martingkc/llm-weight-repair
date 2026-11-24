import inspect
for nm in ('exact21','h_and_x','getBankLayer','addroute','install_banks','mgate19','gate19v','topread','put','mlpcoord','topdiag','spectrum','bitstats','install_routes','add23','path23'):
    try: print(nm,inspect.getsource(globals()[nm]))
    except: pass
print(sorted(os.listdir('.'))); print(edited())