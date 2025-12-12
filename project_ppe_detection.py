

import argparse
import time
from collections import defaultdict
import csv

import cv2
import numpy as np
from ultralytics import YOLO


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_over_person(person_box, item_box):
    # compute fraction of person bbox covered by item bbox (intersection area / person area)
    x1 = max(person_box[0], item_box[0])
    y1 = max(person_box[1], item_box[1])
    x2 = min(person_box[2], item_box[2])
    y2 = min(person_box[3], item_box[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    person_a = box_area(person_box)
    if person_a <= 0:
        return 0.0
    return inter / person_a


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="path to input video (or camera index as int)")
    parser.add_argument("--model", required=True, help="path to YOLO weights or model name (e.g. yolov8n.pt or best.pt)")
    parser.add_argument("--output", default="output_annotated.mp4", help="path to save annotated video")
    parser.add_argument("--csv", default="summary.csv", help="summary CSV output")
    parser.add_argument("--display", action="store_true", help="show frames in a window while processing")
    parser.add_argument("--device", default="cpu", help="inference device: cpu or 0,1 for gpu")
    parser.add_argument("--helmet_class", default="helmet", help="name of helmet class in your model")
    parser.add_argument("--vest_class", default="vest", help="name of vest class in your model")
    parser.add_argument("--person_class", default="person", help="name of person class in your model")
    parser.add_argument("--coverage_thresh", type=float, default=0.10, help="min coverage fraction to consider an item belongs to a person")
    return parser.parse_args()


def main():
    args = parse_args()

    # open model
    print(f"Loading model '{args.model}' on device {args.device} ...")
    model = YOLO(args.model)
    # set model device if specified (ultralytics will route automatically if model string provided)
    # model.to(args.device)

    # get class id for relevant names (if available)
    names = model.names  # dict id->name
    name_to_id = {v: k for k, v in names.items()}
    helmet_id = name_to_id.get(args.helmet_class, None)
    vest_id = name_to_id.get(args.vest_class, None)
    person_id = name_to_id.get(args.person_class, None)

    print("Model classes:", names)
    print(f"Using helmet_id={helmet_id}, vest_id={vest_id}, person_id={person_id}")

    # open video capture
    try:
        idx = int(args.video)
        cap = cv2.VideoCapture(idx)
    except Exception:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_idx = 0
    t0 = time.time()

    # summary counters
    total_persons_seen = 0
    total_person_instances = 0
    person_alerts = 0
    per_frame_counts = []

    # run per-frame detection
    print("Starting processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # run model (ultralytics accepts numpy frames)
        results = model(frame)[0]

        # collect detections
        boxes = []  # list of (xmin, ymin, xmax, ymax, cls_id, conf)
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                boxes.append((x1, y1, x2, y2, cls, conf))

        # separate persons and items
        persons = [b for b in boxes if (person_id is None or b[4] == person_id)]
        helmets = [b for b in boxes if (helmet_id is not None and b[4] == helmet_id)]
        vests = [b for b in boxes if (vest_id is not None and b[4] == vest_id)]

        # fallback: if person_id unknown, treat 'person' class id via name search
        if person_id is None:
            # try to find any class named 'person' or nearest
            for k, v in names.items():
                if v.lower() == 'person':
                    person_id = k
                    persons = [b for b in boxes if b[4] == person_id]
                    break

        frame_person_count = len(persons)
        total_persons_seen += frame_person_count

        # analyze per person
        alerts_in_frame = 0
        for p in persons:
            px1, py1, px2, py2, pcls, pconf = p
            person_box = (px1, py1, px2, py2)
            has_helmet = False
            has_vest = False

            # check helmets
            for hbox in helmets:
                hb = (hbox[0], hbox[1], hbox[2], hbox[3])
                cov = intersection_over_person(person_box, hb)
                if cov >= args.coverage_thresh:
                    has_helmet = True
                    break

            # check vests
            for vbox in vests:
                vb = (vbox[0], vbox[1], vbox[2], vbox[3])
                cov = intersection_over_person(person_box, vb)
                if cov >= args.coverage_thresh:
                    has_vest = True
                    break

            # draw person box
            px1i, py1i, px2i, py2i = map(int, (px1, py1, px2, py2))
            if has_helmet and has_vest:
                color = (0, 255, 0)  # green: safe
                label = "PPE OK"
            else:
                color = (0, 0, 255)  # red: alert
                missing = []
                if not has_helmet:
                    missing.append('Helmet')
                if not has_vest:
                    missing.append('Vest')
                label = "ALERT: Missing " + ",".join(missing)
                person_alerts += 1
                alerts_in_frame += 1

            # person bbox
            cv2.rectangle(frame, (px1i, py1i), (px2i, py2i), color, 2)
            cv2.putText(frame, label, (px1i, max(20, py1i - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw boxes for helmets and vests for visualization
        for hb in helmets:
            x1, y1, x2, y2, cls, conf = hb
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
            name = names.get(cls, str(cls))
            cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        for vb in vests:
            x1, y1, x2, y2, cls, conf = vb
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
            name = names.get(cls, str(cls))
            cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # HUD: fps, frame, counts
        elapsed = time.time() - t0 if t0 else 0.0
        fps_display = frame_idx / elapsed if elapsed > 0 else 0.0
        hud = f"Frame:{frame_idx} FPS:{fps_display:.1f} Persons:{frame_person_count} Alerts:{alerts_in_frame}"
        cv2.putText(frame, hud, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

        # write frame
        out_writer.write(frame)

        if args.display:
            cv2.imshow('PPE Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        per_frame_counts.append({'frame': frame_idx, 'persons': frame_person_count, 'alerts': alerts_in_frame})

    # cleanup
    cap.release()
    out_writer.release()
    if args.display:
        cv2.destroyAllWindows()

    # summary
    total_frames = frame_idx
    total_alerts = person_alerts
    avg_persons_per_frame = total_persons_seen / total_frames if total_frames else 0
    alert_rate = total_alerts / total_persons_seen if total_persons_seen else 0

    print("\n===== Processing complete =====")
    print(f"Frames processed: {total_frames}")
    print(f"Total person detections (per-frame sum): {total_persons_seen}")
    print(f"Total person-alerts (missing PPE): {total_alerts}")
    print(f"Average persons per frame: {avg_persons_per_frame:.2f}")
    print(f"Alert rate (alerts / person_instances): {alert_rate:.3f}")
    print(f"Annotated video saved to: {args.output}")
    print(f"Summary CSV saved to: {args.csv}")

    # save CSV summary
    with open(args.csv, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'persons', 'alerts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_frame_counts:
            writer.writerow(row)


if __name__ == '__main__':
    main()
