import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import csv
from datetime import datetime

def find_intersection(point1, point2, point3, point4, point5):
    A1 = point2[1] - point1[1]
    B1 = point1[0] - point2[0]
    C1 = A1 * point1[0] + B1 * point1[1]

    A2 = point5[1] - point3[1]
    B2 = point3[0] - point5[0]
    C2 = A2 * point3[0] + B2 * point3[1]

    matrix_A = np.array([[A1, B1], [A2, B2]])
    matrix_C = np.array([C1, C2])

    if np.linalg.det(matrix_A) == 0:
        return None

    intersection = np.linalg.solve(matrix_A, matrix_C)
    return (int(intersection[0]), int(intersection[1]))


def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def process_frame(img, model):
    # Resize frame to 960x720
    img = cv2.resize(img, (1280, 720))
    H, W, _ = img.shape

    # Run YOLO model
    results = model(img)
    marker_color1 = (255, 0, 0)
    green_color = (0, 255, 0)

    max_contour_length = 0
    max_contour = None
    segmentation_found = False
    detect1 = time.time()

    for result in results:
        if result.masks is not None:
            for mask in result.masks.data:
                mask_cpu = mask.cpu().numpy()
                mask_cpu = cv2.resize(mask_cpu, (W, H))
                contours, _ = cv2.findContours((mask_cpu * 255).astype('uint8'), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) > max_contour_length:
                        max_contour_length = len(contour)
                        max_contour = contour
                segmentation_found = True

    if not segmentation_found:
        return img, None, None

    detect2 = time.time()
    comp1 = time.time()

    # Generate the red mask for the segmented area
    red_mask = np.zeros_like(img)
    if max_contour is not None:
        cv2.drawContours(red_mask, [max_contour], -1, (0, 255, 0), thickness=cv2.FILLED)

    # Overlay the red mask on the original image
    overlay_image = cv2.addWeighted(img, 1.0, red_mask, 0.5, 0)

    # Calculate boundary points for this contour
    all_boundary_points = []
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        for px in range(x, x + w):
            for py in range(y, y + h):
                if cv2.pointPolygonTest(max_contour, (px, py), False) == 0:
                    all_boundary_points.append((px, py))

    # Create a blank image to display the largest contour and pixel points
    blank_image_for_mask = np.zeros_like(img)
    if max_contour is not None:
        cv2.drawContours(blank_image_for_mask, [max_contour], -1, green_color, thickness=cv2.FILLED)

        all_boundary_points_array = np.array(all_boundary_points)

        # Filter points where y > 360
        boundary_points_sorted = all_boundary_points_array[
            np.lexsort((all_boundary_points_array[:, 0], all_boundary_points_array[:, 1]))]
        filtered_points = boundary_points_sorted[boundary_points_sorted[:, 1] > 360]

        unique_ys = np.unique(filtered_points[:, 1])

        first_valid_pair = None

        for y in unique_ys:
            points_with_y = filtered_points[filtered_points[:, 1] == y]
            valid_points_with_y = points_with_y[points_with_y[:, 0] < 960]

            if valid_points_with_y.size > 0:
                max_x_point = valid_points_with_y[valid_points_with_y[:, 0].argmax()]

                if first_valid_pair is None:
                    first_valid_pair = (max_x_point, y)
                else:
                    break

        if first_valid_pair is not None:
            y_value = first_valid_pair[1]
            points_with_same_y = filtered_points[filtered_points[:, 1] == y_value]
            points_with_same_y_and_x_condition = points_with_same_y[points_with_same_y[:, 0] > 0]

            if points_with_same_y_and_x_condition.size > 0:
                point1 = points_with_same_y_and_x_condition[points_with_same_y_and_x_condition[:, 0].argmin()]
                point2 = first_valid_pair[0]
            else:
                point1, point2 = None, None
        else:
            point1, point2 = None, None

        if point1 is not None and point2 is not None:
            cv2.line(blank_image_for_mask, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])),
                     (0, 0, 255), thickness=2)

            centerCar_intersect = find_intersection(point1, point2, (666, 490), (319, 586), (681, 560))
            if centerCar_intersect is not None:
                cv2.circle(blank_image_for_mask, centerCar_intersect, 5, (0, 255, 255), -1)
                line_center = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
                cv2.circle(blank_image_for_mask, line_center, 5, (255, 255, 0), -1)

                if centerCar_intersect[0] < line_center[0]:
                    direction = "left"
                else:
                    direction = "right"

                len_pix = calculate_distance(point1, point2)
                offset_pix = calculate_distance(centerCar_intersect, line_center)
                offset_m = (offset_pix / len_pix) * 3.6
                comp2 = time.time()

                print(f"{detect2 - detect1:.2f} seconds for mask")
                print(f"{comp2 - comp1:.2f} seconds for calibration")
                print(f"Offset: {offset_m:.4f} m towards {direction}")
                cv2.putText(overlay_image, f"Offset: {offset_m:.4f} m towards {direction}", (460, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                # Draw the specified lines
                cv2.fillPoly(overlay_image,
                             [np.array(((656, 450), (658, 450), (685, 560), (677, 560)), dtype=np.int32)],
                             (255, 255, 255))
                cv2.fillPoly(overlay_image,
                            [np.array(((614, 450), (616, 450), (570, 560), (560, 560)), dtype=np.int32)],
                            (255, 0, 0))

                cv2.fillPoly(overlay_image,
                             [np.array(((709, 450), (711, 450), (809, 560), (799, 560)), dtype=np.int32)],
                             (255, 0, 0))

                return overlay_image, offset_m, direction
        return overlay_image, None, None
    return img, None, None


# Choose input option
input_option = 1  # Change this to 1 for video file or 2 for camera feed

if input_option == 1:
    video_path = r'/home/dev/Documents/Autonomous_Vehicle/Images_dataset/DATA_JUNE/Highwayloop_right_2024_06_13_18_04_14.mp4'
    cap = cv2.VideoCapture(video_path)
elif input_option == 2:
    cap = cv2.VideoCapture(2)  # Use the second camera

# Check if video source successfully opened
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Check if CUDA (GPU) is available and enable it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load YOLO model on the selected device
model_path = r'/home/dev/Documents/Autonomous_Vehicle/Images_dataset/imagesPacificaFinal/runs/weights/last.pt'
model = YOLO(model_path).to(device)

# Define the codec and create VideoWriter object
#output_path = 'testecosedrive.mp4'
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
#fps = 30  # Frames per second
#frame_size = (1280, 720)
#out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Initialize CSV file for recording frame data
#csv_path = 'testecosedrive.csv'
#csv_file = open(csv_path, mode='w', newline='')
#csv_writer = csv.writer(csv_file)
#csv_writer.writerow(['current_datetime ("%Y_%m_%d_%H_%M_%S_%f")', 'Video Time (s)', 'Offset (m)', 'Direction'])

frame_count = 0


# Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if input_option == 2:
        # Rotate frame by 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        # Crop the right half of the frame
        frame = frame[:, frame.shape[1] // 2:]

    # Move frame to device
    frame_tensor = torch.from_numpy(frame).unsqueeze(0).to(device)

    #video_time = frame_count / fps

    current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]

    # Process frame on the model
    processed_frame, offset_m, direction = process_frame(frame, model)

    # Write the processed frame to the output video
    #out.write(processed_frame)

    #if offset_m is not None and direction is not None:
    #    csv_writer.writerow([current_datetime,video_time, offset_m, direction])

    if offset_m is not None:
        print(f"Offset: {offset_m:.4f} m towards {direction}")

    cv2.imshow('Processed Frame', processed_frame)

    # Wait for a small amount of time (e.g., 1 millisecond)
    key = cv2.waitKey(1)

    # Break the loop if 'q' key is pressed
    if key == ord('q'):
        break

    frame_count += 1

# Release video capture object and close OpenCV windows
cap.release()
#out.release()
cv2.destroyAllWindows()
