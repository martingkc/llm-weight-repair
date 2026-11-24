import dis
for k in ['writeG','writeJ','writeH','getblock','clipgate_from','setup24','phase_reader','align_D','lwdir','getp','att_recorded']:
 z=globals()[k]; print(k, z.__code__.co_varnames, z.__code__.co_consts if k not in ['writeG','writeJ'] else tuple(str(c)[:190] for c in z.__code__.co_consts));
for k in ['Fopen','Fcare','Fchs','Fresults','Fidle','Gchs','cd24','dpaired','JD','JK','p15','u15','m19','q19','i19','fr19','s19','c19','t19']:
 z=globals().get(k); print(k, type(z),str(np.shape(z) if isinstance(z,np.ndarray) else str(z)[:900]))