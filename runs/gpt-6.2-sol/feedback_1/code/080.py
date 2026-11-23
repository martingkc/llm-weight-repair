for nmX, rangesX in {'att_weight_contract': [19,20,33,34,35,37], 'cancel_agg':[18,19,20,23,25,29,33,35,36,37]}.items():
 print(nmX)
 print('\n'.join(f'{j0.line_number if j0.starts_line else ""} {j0.opname:20} {j0.argrepr}' for j0 in dis.get_instructions(globals()[nmX]) if j0.line_number in rangesX))