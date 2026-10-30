"""CPU rendering of source + camera overlay + estimated world skeleton.

All output pictures come from recorded video and estimated numerical joints.
This is a visual inspection artifact, not proof of accuracy or physical tracking.
"""
import argparse
from pathlib import Path
import json

import av
import cv2
import numpy as np

PARENTS=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])


def skeleton(image,points,color=(60,220,255)):
    for j,parent in enumerate(PARENTS):
        if parent<0 or not np.isfinite(points[[parent,j]]).all():
            continue
        a,b=[tuple(np.clip(p,-10000,10000).astype(int)) for p in points[[parent,j]]]
        cv2.line(image,a,b,color,2,cv2.LINE_AA)


def render(video,folder,output):
    folder,output=Path(folder),Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan=json.loads((folder/"motion_windows.json").read_text())
    source=cv2.VideoCapture(str(video))
    if abs(source.get(cv2.CAP_PROP_FPS)-30)>.001:
        raise ValueError("Use the actual 30fps normalized inference input")
    width,height=640,360
    panels=3
    destination=av.open(str(output),mode="w")
    encoder=destination.add_stream("libx264",rate=15)
    encoder.width=width*panels;encoder.height=height;encoder.pix_fmt="yuv420p"
    encoder.options={"crf":"22","preset":"fast"}
    frame_index,encoded=0,0
    expected=plan["windows"][-1]["core_end"]
    try:
        for window in plan["windows"]:
            with np.load(folder/"motion_windows"/f"window_{window['index']:05d}_joints.npz") as data:
                incam,world,K=data["joints_incam"],data["joints_global"],data["K"]
            # Fixed scale within an inference window; not frame-wise zoom.
            xy=world[...,[0,1]]
            relative=xy-xy[:,[0]]
            span=max(float(np.ptp(relative,axis=1).max()),1.)
            for i in range(window["core_start"],window["core_end"]):
                if frame_index!=i:
                    raise ValueError("Noncontiguous review ownership")
                valid,image=source.read()
                if not valid:
                    raise ValueError("Video ended before numerical sequence")
                frame_index+=1
                if i%2:
                    continue
                k=i-window["input_start"]
                h,w=image.shape[:2]
                # Letterbox, preserving source aspect ratio.
                scale=min(width/w,(height-35)/h)
                small=cv2.resize(image,(round(w*scale),round(h*scale)))
                original=np.zeros((height,width,3),np.uint8)
                x=(width-small.shape[1])//2;y=35+(height-35-small.shape[0])//2
                original[y:y+small.shape[0],x:x+small.shape[1]]=small
                overlay=original.copy()
                p=incam[k]
                projected=(p@K[k].T)
                projected=projected[:,:2]/projected[:,2:3]
                projected=projected*scale+np.array([x,y])
                skeleton(overlay,projected)
                world_image=np.full_like(original,30)
                # Root centering is display-only; saved world motion is untouched.
                world_points=relative[k]*(280/span)+np.array([width/2,height/2])
                # GVHMR global is y-up, so flip image vertical coordinate.
                world_points[:,1]=height-world_points[:,1]
                skeleton(world_image,world_points)
                for panel,title in ((original,"Source"),(overlay,"Estimated camera skeleton"),
                                     (world_image,"World pose: root-centered display only")):
                    cv2.putText(panel,title,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
                canvas=np.concatenate([original,overlay,world_image],axis=1)
                cv2.putText(canvas,f"{folder.name} frame={i} t={i/30:.3f}s window={window['index']}",
                            (8,height-8),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                frame=av.VideoFrame.from_ndarray(canvas,format="bgr24")
                for packet in encoder.encode(frame):
                    destination.mux(packet)
                encoded+=1
        for packet in encoder.encode():
            destination.mux(packet)
    finally:
        source.release();destination.close()
    if frame_index!=expected:
        raise ValueError("Incomplete review video")
    result=dict(frames=encoded,fps=15,source_frames=frame_index,output=str(output),
                kind="estimated_human_not_G1",visual_inspection_performed=False,
                audio_included=False,panels=panels,source_video=str(video),
                world_panel="root-centered for display; stored translation retained")
    output.with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--video",required=True)
    parser.add_argument("--folder",required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args()
    render(a.video,a.folder,a.output)
