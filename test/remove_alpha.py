
import numpy as np






def mask_onehot(mask, num_classes):
    _mask = [mask ==i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot_mask(mask):
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask