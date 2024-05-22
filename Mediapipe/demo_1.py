import mediapipe as mp
import cv2
import plot_pose_live
import matplotlib.pyplot as plt

# setup plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# setup mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# open webcam
video_path ='ym.mp4'
# cap = cv2.VideoCapture(1)  # change index
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(
        min_tracking_confidence=0.5,
        min_detection_confidence=0.5,
        model_complexity=1,
        smooth_landmarks=True,
) as pose:
    while cap.isOpened():
        # read webcam image
        success, image = cap.read()

        # skip empty frames
        if not success:
            continue

        # calculate pose
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # draw 3D pose landmarks live
        plot_pose_live.plot_world_landmarks(ax, results.pose_world_landmarks)

        # draw image
        cv2.imshow("MediaPipePose", cv2.flip(image, 1))
        # cv2.waitKey(0)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()