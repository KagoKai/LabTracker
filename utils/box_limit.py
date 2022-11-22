def limited_box(x1, y1, x2, y2, width = 640, height = 480):
    '''
        Return bbox limited to the pixel size of the image
        Output: [left, top, right, bottom]
    '''
    x1_limited = max(min(x1, width), 0)
    y1_limited = max(min(y1, height), 0)
    x2_limited = max(min(x2, width), 0)
    y2_limited = max(min(y2, height), 0)
    return (x1_limited, y1_limited, x2_limited, y2_limited)