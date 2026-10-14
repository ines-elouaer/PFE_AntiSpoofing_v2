import os
import cv2


def load_dnn(model_dir="models"):
    proto = os.path.join(model_dir, "deploy.prototxt")
    model = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(proto):
        raise FileNotFoundError(f"Missing: {proto}")
    if not os.path.exists(model):
        raise FileNotFoundError(f"Missing: {model}")

    net = cv2.dnn.readNetFromCaffe(proto, model)
    return net


def detect_face_dnn(net, img_bgr, conf_thresh=0.5):
    
    h, w = img_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img_bgr, 1.0, (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False, crop=False
    )

    net.setInput(blob)
    det = net.forward()

    best = None
    best_conf = conf_thresh

    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < best_conf:
            continue

        x1 = int(det[0, 0, i, 3] * w)
        y1 = int(det[0, 0, i, 4] * h)
        x2 = int(det[0, 0, i, 5] * w)
        y2 = int(det[0, 0, i, 6] * h)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw == 0 or bh == 0:
            continue

        best = (x1, y1, bw, bh, conf)
        best_conf = conf

    if best is None:
        return None, None

    x, y, bw, bh, conf = best
    return (x, y, bw, bh), conf
