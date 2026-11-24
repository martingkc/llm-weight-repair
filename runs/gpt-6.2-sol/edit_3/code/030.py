for jj in range(24):
 print(jj, ' '.join(f'{metrics[jj,m][0]:.4f}:{metrics[jj,m][3][0]:.2f}/{metrics[jj,m][3][7]:.2f}/{metrics[jj,m][3][-1]:.2f}' for m in mods))
print(mods); print('edits',edited());print('control questions last medical',PRACTICE['medical'][-1]);
print('practice medical range %',[(ii, len(tok.encode(q['reference']).ids)) for ii,q in enumerate(PRACTICE['medical'])][::5])