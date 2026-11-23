print(CONFIG)
print(len(NAMES),NAMES[:23],NAMES[-10:])
print(len(ADAPTED),list(ADAPTED)[:20])
for m in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
 a,b,s=lora('model.layers.0.'+m); print(m,a.shape,b.shape,s,a.dtype,b.dtype,np.linalg.norm(a),np.linalg.norm(b), np.linalg.norm(a.mean(axis=1)), np.linalg.norm(b.mean(axis=0)))
import os
print(os.environ.get('OPENBLAS_NUM_THREADS')); print(np.__version__)