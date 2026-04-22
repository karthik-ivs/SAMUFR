import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import cv2
from PIL import Image
from ultralytics import YOLO
from collections import deque
import pandas as pd
from datetime import datetime
import os

# ================================
# PERIOD SCHEDULE
# ================================
PERIOD_SCHEDULE = [
    ("Period 1", "08:15", "09:05"),
    ("Period 2", "09:05", "09:55"),
    ("Period 3", "10:10", "11:00"),
    ("Period 4", "11:00", "11:50"),
    ("Period 5", "11:50", "12:40"),
    ("Period 6", "13:40", "14:30"),
    ("Period 7", "14:30", "15:20"),
    ("Period 8", "15:20", "16:30")
]

def get_current_period():
    now = datetime.now().time()

    for period_name, start, end in PERIOD_SCHEDULE:
        start_time = datetime.strptime(start, "%H:%M").time()
        end_time = datetime.strptime(end, "%H:%M").time()

        if start_time <= now <= end_time:
            return period_name

    return None

# ================================
# EXCEL FUNCTION (DATE → SHEET)
# ================================
excel_file = "attendance.xlsx"

def mark_attendance_excel(name, date, period):

    # Create file if not exists
    if not os.path.exists(excel_file):
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            pd.DataFrame(columns=["Period", "Name"]).to_excel(
                writer, sheet_name=date, index=False
            )

    # Load all sheets
    try:
        existing_sheets = pd.read_excel(excel_file, sheet_name=None)
    except:
        existing_sheets = {}

    # Get today's sheet
    if date in existing_sheets:
        df = existing_sheets[date]
    else:
        df = pd.DataFrame(columns=["Period", "Name"])

    # Avoid duplicate (same name + period)
    already_marked = (
        (df["Name"] == name) &
        (df["Period"] == period)
    ).any()

    if not already_marked:
        new_entry = pd.DataFrame([{
            "Period": period,
            "Name": name
        }])

        df = pd.concat([df, new_entry], ignore_index=True)
        existing_sheets[date] = df

        # Rewrite full Excel
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for sheet, data in existing_sheets.items():
                data.to_excel(writer, sheet_name=sheet, index=False)

        print(f"Attendance Marked: {name} | {period}")

# ================================
# DEVICE
# ================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================================
# LOAD MODEL
# ================================
MODEL_PATH = "best_mbnet_model.pth"

checkpoint = torch.load(MODEL_PATH, map_location=device) # Saves the model after each epoch, but only the best one is saved as "best_mbnet_model.pth"
class_names = checkpoint["class_names"]

model = mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

print("Model loaded successfully")

# ================================
# TRANSFORMS
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ================================
# YOLO FACE DETECTOR
# ================================
yolo_model = YOLO("yolov8n-face.pt")

# ================================
# VIDEO CAPTURE
# ================================
cap = cv2.VideoCapture(0)

# ================================
# MEMORY
# ================================
pred_queue = deque(maxlen=10)
marked_students = set()  # (name, date, period)

# ================================
# DETECTION LOOP
# ================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    current_period = get_current_period()

    # If not in class time
    if current_period is None:
        cv2.putText(frame, "No Active Period", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    results = yolo_model(frame)
    h_frame, w_frame, _ = frame.shape

    for box in results[0].boxes.xyxy:

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)

        if (x2 - x1) < 80 or (y2 - y1) < 80:
            continue

        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w_frame, x2 + margin)
        y2 = min(h_frame, y2 + margin)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face = cv2.resize(face, (224, 224))
        img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            idx = torch.argmax(probs).item()
            confidence = probs[0, idx].item() * 100

        current_name = class_names[idx]
        pred_queue.append((current_name, confidence))

        valid_preds = [p for p in pred_queue if p[1] > 70]

        if len(valid_preds) >= 5:
            names = [p[0] for p in valid_preds]
            final_name = max(set(names), key=names.count)

            display_name = f"{final_name} ({current_period})"

            # ================================
            # MARK ATTENDANCE
            # ================================
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")

            student_key = (final_name, date, current_period)

            if student_key not in marked_students:
                mark_attendance_excel(final_name, date, current_period)
                marked_students.add(student_key)

        else:
            display_name = "Detecting..."

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, display_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()