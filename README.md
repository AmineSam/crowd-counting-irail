# CV Add-on: Head Detection + Crowd Counting (RPEE-Heads + YOLOv8n)

## Project goal
Build a computer-vision component that estimates **crowd occupancy** by detecting **heads** in station-like scenes.

This is an add-on to the [iRail Azure project](https://github.com/AmineSam/challenge-azure): iRail does not provide station occupancy, so we approximate it from camera frames (or public station videos) using deep learning.

[Demo](https://huggingface.co/spaces/AmineSam/irail-crowd-counting-yolov8n-demo) 
<img width="2383" height="1181" alt="web_app" src="https://github.com/user-attachments/assets/e6ff3be0-684a-43a5-a4a3-60a3ec21aebe" />


## Dataset
**RPEE-Heads** (Railway Platforms and Event Entrances-Heads)
- 1,886 images
- 109,913 head annotations
- Split: train (70%), val (15%), test (15%)
- License: CC BY-SA 4.0 (attribution required)
- [Link](http://ped.fz-juelich.de/da/2024rpee_heads) 


## Approach
1. Convert dataset into YOLO format (already provided) and generate a correct `data.yaml` by auto-detecting the nested folder structure.
2. Fine-tune **YOLOv8n** for a single class: `head`.
3. Evaluate detection with YOLO metrics (mAP50, mAP50-95).
4. Convert detections into a **count per frame** and evaluate counting quality using MAE/RMSE/bias and density buckets.
5. Tune inference thresholds (`conf`, `iou`) for counting performance.
6. Analyze harder cases (dense crowds, small heads) and choosing the best conf and ioufor that.

## Key training settings
- Model: YOLOv8n (Ultralytics)
- Input size: `imgsz=832` (larger than default 640 to preserve small heads)
- Epochs: 80
- Batch: 16

## Baseline training metrics
1. Validation set (RPEE-Heads validation)

    - Images: 246

    - Head instances (GT boxes): 16,022

    - Precision (Box P): 0.910

    - Recall (Box R): 0.805

    - mAP@0.50: 0.881

    - mAP@0.50:0.95: 0.522

2. Test set (RPEE-Heads testing)

    - Images: 294

    - Head instances (GT boxes): 15,285

    - Precision (Box P): 0.908

    - Recall (Box R): 0.803

    - mAP@0.50: 0.878

    - mAP@0.50:0.95: 0.515

3. The model generalizes well: validation and test metrics are almost identical, meaning the detector’s performance is stable across unseen data (no obvious overfitting at the detection level).

## Counting results

For each image/frame:

- GT count = number of labeled head boxes in the YOLO label file
- Pred count = number of predicted boxes after confidence filtering + NMS
- Error = pred - gt

We report:

- MAE: average absolute error ("on average we miss/overcount by X heads")
- RMSE: penalizes large errors more strongly
- Bias: average signed error (negative = undercount, positive = overcount)

We also compute bucketed errors by crowd density (0–10, 11–30, …).
Using `conf=0.25`, `iou=0.70`, `imgsz=832`, `max_det=300`:
- Validation

| gt_bucket | n   | mae       | rmse      | bias      |
|-----------|-----|-----------|-----------|-----------|
| 11-30     | 14  | 1.714286  | 2.329929  | -0.714286 |
| 31-60     | 129 | 3.651163  | 5.475102  | -1.077519 |
| 61-100    | 79  | 4.202532  | 6.298483  | -0.911392 |
| 101-200   | 24  | 10.291667 | 18.602195 | -5.208333 |

-Test

| gt_bucket | n   | mae       | rmse      | bias       |
|-----------|-----|-----------|-----------|------------|
| 0-10      | 4   | 1.000000  | 1.224745  | 0.000000   |
| 11-30     | 62  | 1.774194  | 2.416342  | -0.096774  |
| 31-60     | 153 | 2.633987  | 3.609175  | -0.294118  |
| 61-100    | 59  | 4.644068  | 7.378393  | -2.135593  |
| 101-200   | 13  | 16.769231 | 25.374124 | -10.769231 |
| 200+      | 3   | 6.666667  | 7.164728  | 6.666667   |


Bucket analysis shows errors increase with higher crowd density (101-200 heads is the hardest).

<img width="572" height="454" alt="scatter_test" src="https://github.com/user-attachments/assets/b9ee503e-d385-420b-a084-461fdced4bf0" />


<img width="563" height="454" alt="error_hist_test" src="https://github.com/user-attachments/assets/a9d5d048-7763-4f7e-bcfd-476b908ebea5" />

## Tune conf and IoU for best bias and MAE

- Conf:

| conf_idx  | conf_threshold  | mae       | rmse      | bias       |
|------|-----------|-----------|-----------|------------|
| 4    | 0.25      | 4.365854  | 7.907494  | -1.406504  |
| 3    | 0.20      | 5.178862  | 7.670554  | 3.032520   |
| 5    | 0.30      | 5.910569  | 10.491189 | -4.772358  |
| 6    | 0.35      | 8.077236  | 13.255617 | -7.548780  |
| 2    | 0.15      | 9.626016  | 12.153155 | 9.113821   |
| 1    | 0.10      | 18.609756 | 22.810103 | 18.528455  |
| 0    | 0.05      | 37.296748 | 44.588042 | 37.296748  |

- IoU:

| iou_idx  | IoU  | conf_threshold | mae       | rmse      | bias       |
|-----|------|-----------|-----------|-----------|------------|
| 5   | 0.70 | 0.25      | 4.365854  | 7.907494  | -1.406504  |
| 4   | 0.65 | 0.25      | 4.455285  | 8.293400  | -2.341463  |
| 3   | 0.60 | 0.25      | 4.646341  | 8.607286  | -3.020325  |
| 6   | 0.75 | 0.25      | 4.674797  | 8.007111  | 0.097561   |
| 2   | 0.55 | 0.25      | 4.869919  | 8.904644  | -3.422764  |
| 1   | 0.50 | 0.25      | 5.060976  | 9.190619  | -3.735772  |
| 0   | 0.45 | 0.25      | 5.150407  | 9.345013  | -3.930894  |

We are optimizing the model towards bias so best tuning here will be (IoU = 0.75, conf = 0.25):

A detection is counted only if:

- its predicted bounding box overlaps the ground-truth box by at least 75%, and

- the model’s confidence score is at least 0.25

Results of this tuning show :

- Almost zero bias → excellent for fairness and downstream models

- Slightly worse RMSE than looser IoU

- Very stable behavior


## Run the model (local or Kaggle)

### 1) Install dependencies

```bash
pip install -U ultralytics opencv-python pillow numpy
```

### 2) Download the model weights

* If you trained in Kaggle: download `best.pt` from:
  `runs/detect/train3/weights/best.pt`
* If you used the model from Hugging Face: download `best.pt` from your HF **Model repo**.

### 3) Run inference on a single image

```python
from ultralytics import YOLO
import cv2

MODEL_PATH = "best.pt"  # or a local path to your weights
model = YOLO(MODEL_PATH)

img_path = "example.jpg"
img = cv2.imread(img_path)

res = model.predict(
    img,
    imgsz=832,
    conf=0.25,
    iou=0.75,
    max_det=300,
    verbose=False
)[0]

head_count = 0 if res.boxes is None else int(res.boxes.shape[0])
print("Predicted head count:", head_count)

# Save visualization
vis = res.plot()                 # BGR image
cv2.imwrite("pred_overlay.jpg", vis)
```

### 4) Run inference on a video (count per frame)

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

cap = cv2.VideoCapture("video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 25.0,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ok, frame = cap.read()
    if not ok:
        break

    res = model.predict(frame, imgsz=832, conf=0.25, iou=0.70, max_det=300, verbose=False)[0]
    count = 0 if res.boxes is None else int(res.boxes.shape[0])

    overlay = res.plot()
    cv2.putText(overlay, f"Heads: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    out.write(overlay)

cap.release()
out.release()
print("Saved: output.mp4")
```

### Recommended inference settings (baseline)

* `imgsz=832` improves small-head detection (slower than the default 640)
* `conf=0.25`, `iou=0.75`, `max_det=300` were used for the best counting behavior

---

## Use the Hugging Face demo (Space)

### Option A: Use it in the browser

1. Open the Hugging Face **Space** page ([Demo](https://huggingface.co/spaces/AmineSam/irail-crowd-counting-yolov8n-demo) ).
2. Upload an image (or a frame screenshot).
3. Click **Run**.
4. The demo returns:

   * An output image with bounding boxes
   * Predicted head count
   * A simple occupancy bucket (Low / Medium / High)
   * You can adjust the Confidence and the IoU to see the effect on counting

### Option B: Duplicate the Space to your account 

1. Click **Duplicate Space** on Hugging Face.
2. Choose a name under your account.
3. Your duplicated Space will build automatically and you can customize:

   * default `imgsz/conf/iou`
   * bucket thresholds
   * UI text and outputs

### Option C: Run the Space locally

```bash
git clone https://huggingface.co/spaces/AmineSam/irail-crowd-counting-yolov8n
cd irail-crowd-counting-yolov8n
pip install -r requirements.txt
python app.py
```

### Notes

* The demo loads weights from the HF **Model repo** (so you can update the model without editing the Space).
* Counts are an approximation: accuracy depends on camera angle, lighting, resolution, and crowd density.


## Next steps
- Try tiled inference for dense scenes (2x2 tiles).
- Compare YOLOv7-tiny vs YOLOv8n.
- Collect a small sample of target camera frames and fine-tune (domain adaptation).


## Contact / Context

This dataset is part of a BeCode Bootcamp

For questions, reuse, or collaboration, feel free to reach out.

---

## 👤 Author

**Amine Samoudi**
- Linkedin: [@AmineSam](https://www.linkedin.com/in/samoudi/)

---
