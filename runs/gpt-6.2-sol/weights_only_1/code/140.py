print('gaussian usage');compact_dis(gaussian_edit,0)
print('\nassemble stats'); print(inspect.signature(edit_family));compact_dis(edit_family,0)
print('\nmean axis values');print(np.round(axmu[:,[62,208,262,570,490,867]],3));print('ni, nm med',np.round(np.median(ni,axis=1),2),np.round(np.median(nm,axis=1),2))