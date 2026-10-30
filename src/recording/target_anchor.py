"""Choose an existing official track from a visually specified image point."""
def select_anchor_track(ids, boxes, ordered, anchor, width, height):
    import numpy as np
    if not anchor:return list(ordered),None
    x=float(anchor['x'])*width;y=float(anchor['y'])*height;frame=int(anchor['frame'])
    if not 0<=anchor['x']<=1 or not 0<=anchor['y']<=1 or frame<0:
        raise ValueError('Invalid normalized target anchor')
    candidates=[]
    for key in ordered:
        frames=np.asarray(ids[key]);b=np.asarray(boxes[key].cpu() if hasattr(boxes[key],'cpu') else boxes[key])
        hits=np.flatnonzero(frames==frame)
        if not len(hits):continue
        lo_x,lo_y,hi_x,hi_y=b[int(hits[0])]
        if lo_x<=x<=hi_x and lo_y<=y<=hi_y:candidates.append(key)
    if len(candidates)!=1:
        raise ValueError(f'Anchor has {len(candidates)} matching tracks; explicit identity review required')
    chosen=candidates[0]
    return [chosen]+[k for k in ordered if k!=chosen],dict(
        selected_track_id=int(chosen),anchor=anchor,matched_at_input_frame=frame,
        method='human source-frame point inside exactly one official tracking box',
        identity_consistency_across_clip_verified=False)
