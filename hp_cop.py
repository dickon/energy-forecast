
# from ecodan data import heatlossjs
a2w_cop = {
    "min": {
        "data": [
          #-15,  -10,   -7,    2,    7,   12,   15,   20
          [None, 3.64, 3.52, 4.16, 5.69, 6.59, 7.06, 7.78], # 25
          [2.66, 3.01, 2.99, 3.59, 4.64, 5.26, 5.64, 6.26], # 35
          [2.38, 2.68, 2.67, 3.23, 4.03, 4.49, 4.78, 5.25], #  40
          [2.10, 2.34, 2.35, 2.86, 3.41, 3.73, 3.91, 4.23], #  45
          [None, 2.10, 2.12, 2.54, 3.07, 3.32, 3.46, 3.71], #  50
          [None, 1.86, 1.89, 2.21, 2.73, 2.91, 3.01, 3.19], #  55
          [None, None, None, None, None, None, None, None]  # 60
        ]
      },
      "mid": {
        "data": [
          # -15,  -10,   -7,    2,    7,   12,   15,   20
          [None, 3.64, 3.85, 4.90, 5.89, 6.58, 7.08, 7.98], # 25
          [2.66, 3.01, 3.25, 3.54, 4.63, 5.35, 5.79, 6.54], # 35
          [2.38, 2.68, 2.87, 3.35, 4.18, 4.66, 4.97, 5.48], # 40
          [2.10, 2.34, 2.50, 3.15, 3.73, 3.98, 4.15, 4.43], #  45
          [None, 2.10, 2.25, 2.78, 3.23, 3.43, 3.56, 3.78], #  50
          [None, 1.86, 2.00, 2.41, 2.74, 2.88, 2.98, 3.14], #  55
          [None, None, 1.70, 2.05, 2.56, 2.59, 2.62, 2.68]  # 60 (note 1.7 at -7 is not from datasheet)
        ]
      },
      "max": {
        "data": [
          # -15,  -10,   -7,    2,    7,   12,   15,   20
          [None, 3.30, 3.60, 4.20, 5.48, 6.20, 6.65, 7.41], #  25
          [2.44, 2.78, 3.00, 3.50, 4.50, 4.98, 5.28, 5.79], #  35
          [2.22, 2.51, 2.70, 3.15, 4.01, 4.37, 4.59, 4.98], #  40
          [2.00, 2.25, 2.40, 2.80, 3.52, 3.75, 3.91, 4.16], #  45
          [None, 2.05, 2.16, 2.47, 3.10, 3.27, 3.38, 3.57], #  50
          [None, 1.85, 1.92, 2.13, 2.68, 2.78, 2.85, 2.97], #  55
          [None, None, None, 1.80, 2.26, 2.30, 2.33, 2.38]  #  60
        ]
      }
}

def cop(load_ratio=0.8, flow_temperature=40, outside_temperature=2, verbose=False):
    """Work out interpolated coefficent of performance for given parameters
    
    >>> cop(0.8, 40, 2)
    3.15
    >>> cop(0.8, 42.5, 7)
    3.8874999999999997
    >>> cop(0.7, 42.5, 10)
    4.2115

    Unsupporter operational conditions throwV
    >>> cop(0.9, 60, -15)
    Traceback (most recent call last):
    ...
    ValueError: out of bounds
    """
    if load_ratio >= 0.6 and load_ratio < 0.8:
        load_index = 'mid'
    elif load_ratio >= 0.8:
        load_index = 'max'
    else:
        load_ratio = 'min'
    if verbose:
        print('load', load_ratio, 'using table', load_index)
    rec = a2w_cop[load_index]['data']

    rows = [25,35,40,45,50,55,60]
    if flow_temperature <= rows[0]:
        print('WARNING: flow temperature lowest than', rows[0]) 
    row_index = 0
    for i, temp in enumerate(rows):
        if i == len(rows)-1 or rows[i+1]>flow_temperature:
            row_index = i 
            if verbose:
                print('flow stick at',row_index)
            break
    assert row_index < len(rec)
    cols = [-15,-10,-7,2,7,12,15,20]
    col_index = 0
    for j, temp in enumerate(cols):
        if j == len(cols)-1 or cols[j+1]>outside_temperature:
            col_index = j
            if verbose:
                print('outside stick at', col_index)
            break
    if verbose:
        print(f'selected row={row_index} of {len(rows)} col={col_index} of {len(cols)}' )    
    a = rec[row_index][col_index]
    if a is None:
        raise ValueError('out of bounds')
    if col_index+1 < len(cols):
        b = rec[row_index][col_index+1]
        c1 = a + (b-a) * (outside_temperature - cols[col_index]) / (cols[col_index+1] - cols[col_index])
    else:
        c1 = a
    a = rec[row_index][col_index]
    if row_index < len(rows)+1:
        b = rec[row_index+1][col_index]
        c2 = a + (b-a) * (flow_temperature - rows[row_index]) / (rows[row_index+1]- rows[row_index])
    else:
        c2 = b
    return c1 + (c2-c1) * (flow_temperature - rows[row_index]) / (rows[row_index+1]-rows[row_index])
    
