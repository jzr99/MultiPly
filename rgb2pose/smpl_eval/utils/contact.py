import numpy as np

def contact_dist(contact_label, pred0, pred1):
    error_contact_dist = []
    if contact_label[0].any():
        error_contact_dist.append(np.sqrt(((pred0[contact_label[0]>0]- pred1[contact_label[0][contact_label[0]>0].astype(int)-1]) ** 2).sum(-1)))
    if contact_label[1].any():
        error_contact_dist.append(np.sqrt(((pred1[contact_label[1]>0]- pred0[contact_label[1][contact_label[1]>0].astype(int)-1]) ** 2).sum(-1)))
    
    return error_contact_dist