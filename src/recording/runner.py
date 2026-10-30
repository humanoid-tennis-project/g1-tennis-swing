"""Resident official GVHMR, with bounded crop reuse on continuous recordings.

Official tracking, crop math, ViTPose flip test, HMR2 and GVHMR postprocessing
are retained. Time windows bound memory; their global frames remain explicit.
Independent 3D windows are NOT asserted to share a world origin.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time


class NoTrack(RuntimeError):
    pass


def recording_precision(folder, requested):
    """Never mix new FP16 features into a previously started FP32 recording."""
    if requested not in ('fp32', 'amp_fp16'):
        raise ValueError('Unsupported HMR2 precision')
    folder = Path(folder)
    manifest = folder/'precision.json'
    if manifest.exists():
        value = json.loads(manifest.read_text())['hmr2']
        if value not in ('fp32', 'amp_fp16'):
            raise ValueError('Invalid saved precision')
        return value
    if any((folder/'feature_packs').glob('pack_*.pt')) or (folder/'preprocess/vit_features.pt').exists():
        return 'fp32'
    return requested


def configure(video, output, resume=False):
    import cv2
    video = Path(video).resolve()
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if abs(fps-30) > .001:
        raise ValueError("Input must be truly resampled to 30fps, not relabelled")
    official = runpy.run_path("tools/demo/demo.py", run_name="official_recording_library")
    folder = Path(output)/video.stem
    folder.mkdir(parents=True, exist_ok=resume)
    link = folder/"0_input_video.mp4"
    if not link.is_symlink() and not link.exists():
        link.symlink_to(video)
    if link.resolve() != video:
        raise ValueError("Resume input differs from the original input")
    previous = sys.argv
    sys.argv = ["demo.py", "--video", str(video), "--output_root", str(output), "--static_cam"]
    try:
        cfg = official["parse_args_to_cfg"]()
    finally:
        sys.argv = previous
    return official, cfg


def iter_shared_crops(video, boxes, chunk_frames=64, skip_frames=0):
    """Decode the half-size stream once and use the official crop math.

    Half-size images + half-size boxes through path_type=np/img_ds=1 match the
    official video/img_ds=.5 route. Returned adjusted boxes are full-image units.
    Never load a whole recording's raw images or crops into RAM/VRAM.
    """
    import imageio.v3 as iio
    import numpy as np
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch
    pending, offset = [], 0
    reader = iio.imiter(str(video), plugin="pyav", filter_sequence=[("scale", "iw*0.5:ih*0.5")])
    for frame in reader:
        pending.append(frame)
        if len(pending) == chunk_frames:
            end = offset+len(pending)
            if end <= skip_frames:
                offset = end
                pending.clear()
                continue
            if end > len(boxes):
                raise ValueError("Decoded frame count exceeds tracking")
            crops, adjusted = get_batch(np.stack(pending), boxes[offset:end]*.5,
                                        img_ds=1., path_type="np")
            yield offset, crops, adjusted/.5
            offset = end
            pending.clear()
    if pending:
        end = offset+len(pending)
        if end > skip_frames:
            crops, adjusted = get_batch(np.stack(pending), boxes[offset:end]*.5,
                                        img_ds=1., path_type="np")
            yield offset, crops, adjusted/.5
        offset = end
    if offset != len(boxes):
        raise ValueError(f"Decoded/tracking frame mismatch: {offset}/{len(boxes)}")


class ResidentModels:
    def __init__(self):
        import torch
        self.torch = torch
        self.models = None

    def load(self, official, cfg):
        started = time.monotonic()
        self.models = dict(tracker=official["Tracker"](), pose=official["VitPoseExtractor"](),
                           features=official["Extractor"]())
        self.models["tracker"].yolo.to("cuda")
        model = official["hydra"].utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        self.models["motion"] = model.eval().cuda()
        return time.monotonic()-started

    def process(self, video, output, source, chunk_frames=64, core_frames=900, context_frames=60,
                crop_equivalence=False, resume=False, checkpoint=None, hmr2_precision='fp32', target_anchor=None):
        from recording.core import motion_windows, save_new
        torch = self.torch
        started = time.monotonic()
        official, cfg = configure(video, output, resume=resume)
        folder = Path(cfg.output_dir)
        hmr2_precision = recording_precision(folder, hmr2_precision)
        if not (folder/'precision.json').exists():
            save_new(folder/'precision.json', dict(hmr2=hmr2_precision, weights='fp32',
                yolo='unchanged', vitpose='fp32', gvhmr='fp32', autocast_cache=False,
                feature_storage='fp32', boundary='whole recording'))
        source_path = folder/"source.json"
        if source_path.exists():
            if json.loads(source_path.read_text()) != source:
                raise ValueError("Resume source identity mismatch")
        else:
            save_new(source_path, source)
        if (folder/"completion.json").exists():
            previous = json.loads((folder/"completion.json").read_text())
            if previous["status"] in ("completed", "no_human_track"):
                return previous
            (folder/"completion.json").rename(folder/f"completion_failed_{time.time_ns()}.json")
        events = []
        def event(stage, before, **extra):
            torch.cuda.synchronize()
            events.append(dict(stage=stage, wall_s=time.monotonic()-before, **extra))
            (folder/"progress.json").write_text(json.dumps(dict(events=events,
                              elapsed_s=time.monotonic()-started), indent=2))
            if checkpoint is not None and stage in ("tracking", "feature_pack_saved"):
                checkpoint()
        status = "failed"
        try:
            if self.models is None:
                load_s = self.load(official, cfg)
                events.append(dict(stage="models_load", wall_s=load_s))
            tracker = self.models["tracker"]
            predictor = getattr(tracker.yolo, "predictor", None)
            if predictor is not None:
                for item in getattr(predictor, "trackers", []):
                    item.reset()
                if hasattr(predictor, "vid_path"):
                    predictor.vid_path = [None]*len(predictor.vid_path)
            # Observe the official sorting; do not replace detection/selection.
            original_sort = tracker.sort_track_length
            def observed_sort(history, path):
                ids, boxes, ordered = original_sort(history, path)
                selection_evidence=None
                if target_anchor:
                    import cv2
                    from recording.target_anchor import select_anchor_track
                    cap=cv2.VideoCapture(str(cfg.video_path))
                    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH);height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT);cap.release()
                    ordered,selection_evidence=select_anchor_track(ids,boxes,ordered,target_anchor,width,height)
                save_new(folder/"tracking_observations.json", dict(total_frames=len(history),
                    selected_track_id=int(ordered[0]) if ordered else None,
                    tracks=[dict(track_id=int(k), frame_ids=ids[k], bbox_xyxy=boxes[k].tolist()) for k in ordered],
                    selector="human frame anchor" if target_anchor else "official summed bbox area",
                    selection_evidence=selection_evidence, missing_boxes="official interpolation retained"))
                if not ordered:
                    raise NoTrack("No official human track")
                return ids, boxes, ordered
            before = time.monotonic()
            if resume and Path(cfg.paths.bbx).exists():
                cached = torch.load(cfg.paths.bbx, map_location="cpu")
                xyxy, xys = cached["bbx_xyxy"], cached["bbx_xys"]
            else:
                if (folder/"tracking_observations.json").exists():
                    (folder/"tracking_observations.json").rename(folder/f"tracking_previous_{time.time_ns()}.json")
                tracker.sort_track_length = observed_sort
                try:
                    xyxy = tracker.get_one_track(str(cfg.video_path)).float()
                finally:
                    tracker.sort_track_length = original_sort
                xys = official["get_bbx_xys_from_xyxy"](xyxy, base_enlarge=1.2).float()
                torch.save(dict(bbx_xyxy=xyxy, bbx_xys=xys), cfg.paths.bbx)
            event("tracking", before, frames=len(xyxy))
            if crop_equivalence:
                from hmr4d.utils.preproc.vitfeat_extractor import get_batch
                before = time.monotonic()
                expected, expected_boxes = get_batch(str(cfg.video_path), xys, img_ds=.5)
                actual, adjusted = [], []
                for _, images, boxes in iter_shared_crops(cfg.video_path, xys, chunk_frames):
                    actual.append(images)
                    adjusted.append(boxes)
                actual, adjusted = torch.cat(actual), torch.cat(adjusted)
                difference = float((expected-actual).abs().max())
                box_diff = float((expected_boxes-adjusted).abs().max())
                save_new(folder/"crop_equivalence.json", dict(crop_max_abs=difference,
                    adjusted_box_max_abs=box_diff, passed=bool(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
                    and torch.allclose(expected_boxes, adjusted, atol=1e-6, rtol=1e-6))))
                del expected, expected_boxes, actual, adjusted
                if difference > 1e-6 or box_diff > 1e-6:
                    raise RuntimeError("Shared crops differ from official path")
                event("crop_equivalence_diagnostic_only", before)
            before = time.monotonic()
            poses, features, expected_offset = [], [], 0
            packs = folder/"feature_packs"
            packs.mkdir(exist_ok=True)
            # A pack owns up to 960 frames. Completed packs survive interruptions.
            while resume and (packs/f"pack_{expected_offset:08d}.pt").exists():
                pack = torch.load(packs/f"pack_{expected_offset:08d}.pt", map_location="cpu")
                if pack.get('hmr2_precision','fp32') != hmr2_precision:
                    raise ValueError('Feature pack precision mismatch')
                if pack["start"] != expected_offset or len(pack["pose"]) != len(pack["features"]):
                    raise ValueError("Invalid cached feature pack")
                poses.append(pack["pose"]); features.append(pack["features"])
                expected_offset += len(pack["pose"])
            pack_poses, pack_features, pack_start = [], [], expected_offset
            for offset, images, adjusted in iter_shared_crops(cfg.video_path, xys, chunk_frames,
                                                             skip_frames=expected_offset):
                if offset != expected_offset:
                    raise ValueError("Noncontiguous crop stream")
                with torch.inference_mode():
                    pack_poses.append(self.models["pose"].extract(images, adjusted))
                    with torch.autocast('cuda', dtype=torch.float16,
                                        enabled=hmr2_precision=='amp_fp16', cache_enabled=False):
                        value = self.models["features"].extract_video_features(images, adjusted)
                    pack_features.append(value.float())
                expected_offset += len(images)
                del images, adjusted
                if expected_offset-pack_start >= 960 or expected_offset == len(xys):
                    p, f = torch.cat(pack_poses), torch.cat(pack_features)
                    target = packs/f"pack_{pack_start:08d}.pt"
                    temporary = target.with_suffix(f".{time.time_ns()}.partial")
                    torch.save(dict(start=pack_start, pose=p, features=f, hmr2_precision=hmr2_precision), temporary)
                    temporary.rename(target)
                    poses.append(p); features.append(f)
                    pack_poses.clear(); pack_features.clear(); pack_start = expected_offset
                    event("feature_pack_saved", before, processed_frames=expected_offset, total_frames=len(xys))
            pose, feature = torch.cat(poses), torch.cat(features)
            for value, path in ((pose, cfg.paths.vitpose),(feature, cfg.paths.vit_features)):
                path=Path(path)
                if path.exists():
                    try:
                        cached=torch.load(path,map_location="cpu")
                        if cached.shape!=value.shape:
                            raise ValueError("Incomplete consolidated cache")
                    except Exception:
                        path.rename(path.with_suffix(f".incomplete_{time.time_ns()}.pt"))
                if not path.exists():
                    temporary=path.with_suffix(f".{time.time_ns()}.partial")
                    torch.save(value,temporary);temporary.rename(path)
            event("decode_crop_pose_features", before, frames=len(pose), batch=16,
                  crop_chunk_frames=chunk_frames, single_shared_crop=True, hmr2_precision=hmr2_precision)
            del poses, features
            windows = motion_windows(len(pose), core_frames, context_frames)
            parameters = Path(folder/"motion_windows")
            parameters.mkdir(exist_ok=resume)
            # Reuse official input construction, camera model and postprocessing.
            all_data = official["load_data_dict"](cfg)
            before = time.monotonic()
            for row in windows:
                import numpy as np
                pt=parameters/f"window_{row['index']:05d}.pt"
                npz=parameters/f"window_{row['index']:05d}_joints.npz"
                if resume and pt.exists() and npz.exists():
                    try:
                        torch.load(pt,map_location="cpu")
                        with np.load(npz) as saved:
                            if not np.array_equal(saved['frame_indices'],np.arange(row['input_start'],row['input_end'])):
                                raise ValueError("Window cache frame mismatch")
                            if not np.isfinite(saved['joints_global']).all():
                                raise ValueError("Invalid joint cache")
                        continue
                    except Exception:
                        pass
                for existing in (pt,npz):
                    if existing.exists():
                        existing.rename(existing.with_suffix(f".incomplete_{time.time_ns()}{existing.suffix}"))
                lo, hi = row["input_start"], row["input_end"]
                data = {k: (torch.tensor(hi-lo) if k == "length" else v[lo:hi]) for k, v in all_data.items()}
                with torch.inference_mode():
                    predicted = self.models["motion"].predict(data, static_cam=True)
                # Persist useful outputs, not large hidden activations for every frame.
                result = {k: official["detach_to_cpu"](predicted[k]) for k in
                          ("smpl_params_global", "smpl_params_incam", "K_fullimg")}
                result["static_conf_logits"] = predicted["net_outputs"]["static_conf_logits"].detach().cpu()
                torch.save(result, parameters/f"window_{row['index']:05d}.pt")
                import numpy as np
                with torch.inference_mode():
                    endecoder = self.models["motion"].pipeline.endecoder
                    joints = {space: endecoder.fk_v2(**{k:v[None].cuda() for k,v in
                              result[f"smpl_params_{space}"].items()})[0].detach().cpu().numpy()
                              for space in ("incam", "global")}
                np.savez_compressed(parameters/f"window_{row['index']:05d}_joints.npz",
                    joints_incam=joints["incam"], joints_global=joints["global"],
                    K=result["K_fullimg"].numpy(), frame_indices=np.arange(lo,hi),
                    fps=30., core_frame_range=np.array([row["core_start"],row["core_end"]]))
                del data, predicted, result
            if not (folder/"motion_windows.json").exists():
                save_new(folder/"motion_windows.json", dict(windows=windows, fps=30,
                source=source, coordinates="independent window world frames; not stitched/calibrated court",
                core_is_stroke_boundary=False, interpolation="official retained"))
            event("gvhmr", before, windows=len(windows))
            status = "completed"
            return dict(status=status, source=source, frames=len(pose), events=events,
                elapsed_s=time.monotonic()-started, output_folder=str(folder),
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                semantic_review="not_performed", pose_batch=16, feature_batch=16, hmr2_precision=hmr2_precision)
        except NoTrack:
            status = "no_human_track"
            return dict(status=status, source=source, events=events, elapsed_s=time.monotonic()-started)
        finally:
            save_new(folder/"completion.json", dict(status=status, source=source, events=events,
                     hmr2_precision=hmr2_precision,
                     elapsed_s=time.monotonic()-started, official_revision="ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--crop-equivalence", action="store_true")
    args = parser.parse_args()
    result = ResidentModels().process(args.video, args.output,
             json.loads(Path(args.source).read_text()), crop_equivalence=args.crop_equivalence)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
