import numpy as np 

def make_anchor(size = 320):
    anchor_list = []

    scale = 1 
    anchor = [16, 8, 32, 64, 128, 256]
    feature_map = [40,20,10]
    if size == 640:
        scale = 2
        anchor = [scale * e for e in anchor]
        feature_map = [scale * e for e in feature_map]
    
    image_h = 320.0 * scale
    image_w = 320.0 * scale

    def feature_map_anchors(fm_w, fm_h, anchor_size, stride_x, stride_y, offset_x, offset_y):
        ah = anchor_size / image_h
        aw = anchor_size / image_w
        cx = offset_x
        cy = offset_y
        bw = aw
        bh = ah
        base_anchor = []

        for i in range(fm_h):
            for j in range(fm_w):
                x1 = j * stride_y + cy
                x2 = j * stride_x + cx
                x3 = bh
                x4 = bw
                base_anchor.append([x1,x2,x3,x4])
        return base_anchor

    for i, e in enumerate(feature_map):
        stride_x = 1.0 / feature_map[i]
        stride_y = 1.0 / feature_map[i]
        offset_x = 0.5 / feature_map[i]
        offset_y = 0.5 / feature_map[i]
        tmp1 = feature_map_anchors(feature_map[i], feature_map[i], anchor[i*2], stride_x, stride_y, offset_x, offset_y)
        tmp2 = feature_map_anchors(feature_map[i], feature_map[i], anchor[i*2 + 1], stride_x, stride_y, offset_x, offset_y)
        for n in range(len(tmp1)):
            anchor_list.append(tmp1[n])
            anchor_list.append(tmp2[n])

    anchors = np.array(anchor_list, dtype = np.float32)

    anchors[:, 0] = anchors[:, 0] - 0.5 * anchors[:, 2]  
    anchors[:, 1] = anchors[:, 1] - 0.5 * anchors[:, 3] 
    anchors[:, 2] = anchors[:, 0] + 0.5 * anchors[:, 2] 
    anchors[:, 3] = anchors[:, 1] + 0.5 * anchors[:, 3]
    anchors = np.clip(anchors, 0.0, 1.0)

    tmp_h = anchors[:, 2] - anchors[:, 0]
    tmp_w = anchors[:, 3] - anchors[:, 1]
    anchors[:, 0] = anchors[:, 0] + 0.5 * tmp_h
    anchors[:, 1] = anchors[:, 1] + 0.5 * tmp_w 
    anchors[:, 2] = tmp_h[:]
    anchors[:, 3] = tmp_w[:]

    return anchors 

