"""
logic/pipeline.py - النسخة النهائية
"""

from ultralytics import YOLO
import numpy as np
import cv2

from logic.ppe_checker    import check_ppe
from logic.ladder_checker import full_ladder_check
from logic.id_reader      import read_id_from_frame
from tracker.tracker      import worker_tracker
from database.db          import save_log, get_employee_info

PPE_CLASSES = {
    0: "person",     1: "helmet",    2: "no_helmet",
    3: "vest",       4: "no_vest",   5: "goggles",
    6: "no_goggles", 7: "gloves",    8: "no_gloves",
}
ID_CLASSES     = {0: "worker_id_tag"}
LADDER_CLASSES = {0: "ladder"}

# ألوان الرسم
COLOR_SAFE    = (0, 200, 0)      # أخضر
COLOR_DANGER  = (0, 0, 220)      # أحمر
COLOR_UNKNOWN = (180, 180, 0)    # أصفر

import torch
class SafetyPipeline:

    def __init__(
        self,
        ppe_weights:    str   = "weights/ppe_best.pt",
        id_weights:     str   = "weights/id_best.pt",
        ladder_weights: str   = "weights/ladder_best.pt",
        pose_weights:   str   = "yolov8s-pose.pt",
        conf_threshold: float = 0.45,
    ):
        torch.load = lambda *args, **kwargs: torch.serialization.load(*args, **kwargs, weights_only=False)
        print("🔄 Loading models...")
        self.ppe_model    = YOLO(ppe_weights)
        self.id_model     = YOLO(id_weights)
        self.ladder_model = YOLO(ladder_weights)
        self.pose_model   = YOLO(pose_weights)
        self.conf         = conf_threshold
        self.worker_id_map: dict[int, str] = {}
        print("✅ All models loaded")

    def process_frame(self, frame: np.ndarray):
        """
        يرجع: (annotated_frame, result_dict)
        """
        # 1. Detections
        ppe_results    = self.ppe_model(frame, conf=self.conf, verbose=False)[0]
        id_results     = self.id_model(frame, conf=self.conf, verbose=False)[0]
        ladder_results = self.ladder_model(frame, conf=self.conf, verbose=False)[0]

        ppe_detections = ppe_results.boxes
        id_detections  = id_results.boxes
        ladder_boxes   = self._get_class_boxes(ladder_results.boxes, 0)

        # 2. Tracking
        person_boxes = self._get_person_detections(ppe_detections)
        tracks       = worker_tracker.update(person_boxes, frame)

        # 3. Pose (لو في سلم)
        pose_keypoints = None
        if ladder_boxes:
            pose_out = self.pose_model(frame, conf=self.conf, verbose=False)[0]
            if pose_out.keypoints is not None and len(pose_out.keypoints.data) > 0:
                pose_keypoints = pose_out.keypoints.data.cpu().numpy()

        # 4. اشتغل على كل عامل
        workers_output = []
        annotated      = frame.copy()

        for track in tracks:
            tid  = track["track_id"]
            pbox = track["bbox"]

            result = self._process_worker(
                frame=frame, track_id=tid, person_bbox=pbox,
                ppe_dets=ppe_detections, id_dets=id_detections,
                ladder_boxes=ladder_boxes, all_kps=pose_keypoints,
            )
            workers_output.append(result)

            if result["alerts"]:
                self._save_to_db(result)

            # ── رسم على الفريم ──
            annotated = self._draw_worker(annotated, result)

        # رسم السلالم
        for lb in ladder_boxes:
            x1, y1, x2, y2 = [int(v) for v in lb]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(annotated, "Ladder", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

        result_dict = {
            "workers":       workers_output,
            "total_workers": len(workers_output),
            "has_ladder":    len(ladder_boxes) > 0,
        }
        return annotated, result_dict

    def _draw_worker(self, frame, result):
        x1, y1, x2, y2 = [int(v) for v in result["bbox"]]
        color = COLOR_SAFE if result["is_safe"] else COLOR_DANGER

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # اسم الموظف + ID
        label = f"{result['employee_id']}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # تنبيهات PPE
        if result["ppe"]["missing"]:
            missing_txt = "Missing: " + ", ".join(result["ppe"]["missing"])
            cv2.putText(frame, missing_txt, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DANGER, 1)

        # تنبيهات السلم
        if result["ladder"] and result["ladder"]["has_alert"]:
            cv2.putText(frame, "! Ladder Alert", (x1, y2 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

        return frame

    def _process_worker(self, frame, track_id, person_bbox,
                        ppe_dets, id_dets, ladder_boxes, all_kps):

        if track_id not in self.worker_id_map:
            emp_id = self._read_worker_id(frame, id_dets, person_bbox)
            if emp_id:
                self.worker_id_map[track_id] = emp_id
                print(f"🪪 {emp_id} (track={track_id})")

        emp_id   = self.worker_id_map.get(track_id, f"UNKNOWN-{track_id}")
        emp_info = get_employee_info(emp_id)
        ppe      = check_ppe(ppe_dets, person_bbox)

        ladder_data = None
        if ladder_boxes:
            nearest = self._nearest_ladder(person_bbox, ladder_boxes)
            if nearest:
                kps = self._get_person_keypoints(person_bbox, all_kps)
                ladder_data = full_ladder_check(
                    frame=frame, person_bbox=person_bbox,
                    ladder_bbox=nearest, keypoints=kps,
                )

        alerts = []
        if not ppe["compliant"]:
            alerts.append({
                "type":    "PPE_VIOLATION",
                "msg":     f"العامل {emp_id} لا يرتدي: {'، '.join(ppe['missing_ar'])}",
                "missing": ppe["missing"],
            })
        if ladder_data and ladder_data["has_alert"]:
            for msg in ladder_data["alerts"]:
                alerts.append({"type": "LADDER_VIOLATION", "msg": f"{emp_id} - {msg}"})

        return {
            "track_id":      track_id,
            "employee_id":   emp_id,
            "employee_name": emp_info.get("name", "-"),
            "bbox":          person_bbox,
            "ppe":           ppe,
            "ladder":        ladder_data,
            "alerts":        alerts,
            "is_safe":       len(alerts) == 0,
        }

    def _read_worker_id(self, frame, id_dets, person_bbox):
        for det in id_dets:
            if int(det.cls[0]) != 0:
                continue
            badge_box = det.xyxy[0].cpu().numpy().tolist()
            if self._boxes_overlap(person_bbox, badge_box, margin=60):
                return read_id_from_frame(frame, badge_box)
        return None

    def _get_person_detections(self, boxes):
        return [b for b in boxes if int(b.cls[0]) == 0]

    def _get_class_boxes(self, boxes, cls_id):
        return [b.xyxy[0].cpu().numpy().tolist()
                for b in boxes if int(b.cls[0]) == cls_id]

    def _nearest_ladder(self, person_bbox, ladder_boxes):
        px = (person_bbox[0] + person_bbox[2]) / 2
        py = (person_bbox[1] + person_bbox[3]) / 2
        nearest, min_dist = None, float("inf")
        for lb in ladder_boxes:
            d = (((px-(lb[0]+lb[2])/2)**2) + ((py-(lb[1]+lb[3])/2)**2))**0.5
            if d < min_dist and d < 300:
                min_dist, nearest = d, lb
        return nearest

    def _get_person_keypoints(self, person_bbox, all_kps):
        if all_kps is None or len(all_kps) == 0:
            return None
        best_kps, best_iou = None, 0.0
        for kps in all_kps:
            valid = kps[kps[:, 2] > 0.3]
            if len(valid) == 0:
                continue
            iou = self._iou(person_bbox,
                            [valid[:,0].min(), valid[:,1].min(),
                             valid[:,0].max(), valid[:,1].max()])
            if iou > best_iou:
                best_iou, best_kps = iou, kps
        return best_kps if best_iou > 0.2 else None

    @staticmethod
    def _boxes_overlap(b1, b2, margin=40):
        return (b1[0]-margin < b2[2] and b1[2]+margin > b2[0] and
                b1[1]-margin < b2[3] and b1[3]+margin > b2[1])

    @staticmethod
    def _iou(A, B):
        xA,yA = max(A[0],B[0]), max(A[1],B[1])
        xB,yB = min(A[2],B[2]), min(A[3],B[3])
        inter = max(0,xB-xA)*max(0,yB-yA)
        if inter == 0: return 0.0
        return inter/((A[2]-A[0])*(A[3]-A[1])+(B[2]-B[0])*(B[3]-B[1])-inter)

    def _save_to_db(self, r):
        ppe  = r["ppe"]
        lad  = r.get("ladder") or {}
        tp   = lad.get("three_point") or {}
        save_log({
            "employee_id":    r["employee_id"],
            "helmet":         "helmet"  in ppe["present"],
            "vest":           "vest"    in ppe["present"],
            "gloves":         "gloves"  in ppe["present"],
            "goggles":        "goggles" in ppe["present"],
            "boots":          False,
            "ppe_compliant":  ppe["compliant"],
            "near_ladder":    lad.get("zone") is not None,
            "ladder_angle":   lad.get("angle"),
            "ladder_zone":    lad.get("zone"),
            "three_point_ok": tp.get("safe"),
            "contact_count":  tp.get("label"),
            "alert_sent":     True,
            "alert_msg":      " | ".join(a["msg"] for a in r["alerts"]),
        })


pipeline = SafetyPipeline()
