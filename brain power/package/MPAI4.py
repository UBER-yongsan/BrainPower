import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
import random
import matplotlib.pyplot as plt

class PoseExtractor:
    def __init__(self, min_detection_confidence=0.7):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=min_detection_confidence)

    def extract_pose_landmarks(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            h, w = image.shape[:2]
            landmarks_2d = np.array(
                [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]
            )
            return landmarks_2d
        return np.empty((0, 2), dtype=int)

def resize_to_same_resolution(img1, img2):
    """Resize img1 and img2 to have the same resolution based on the smaller dimensions."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    new_width = min(w1, w2)
    new_height = min(h1, h2)
    
    resized_img1 = cv2.resize(img1, (new_width, new_height))
    resized_img2 = cv2.resize(img2, (new_width, new_height))
    
    return resized_img1, resized_img2

def sample_points(B, sample_ratio=0.1):
    if len(B) == 0:
        return np.empty((0, 2), dtype=int)
    sample_size = max(1, int(len(B) * sample_ratio))
    return B[np.random.choice(B.shape[0], sample_size, replace=False)]

def is_b_within_a(A, B_sample, threshold=0.9):
    height, width = A.shape
    B_sample = B_sample[(B_sample[:, 0] < width) & (B_sample[:, 1] < height)]
    
    if len(B_sample) == 0:
        return False

    inside_count = np.sum(A[B_sample[:, 1], B_sample[:, 0]] == 255)
    return inside_count / len(B_sample) >= threshold


def process_image(A, landmarks, segmenter_options, image_path):
    mp_image = mp.Image.create_from_file(image_path)
    
    # 여러 개의 ROI를 만들기 위한 리스트
    rois = []
    
    # 랜드마크 중에서 첫 5개의 keypoint를 사용하여 ROI를 만듦 (원하는 대로 조정 가능)
    for i in range(min(5, len(landmarks))):  # 최대 5개까지 ROI를 만듦
        keypoint = containers.keypoint.NormalizedKeypoint(landmarks[i][0], landmarks[i][1])
        roi = vision.InteractiveSegmenterRegionOfInterest(
            format=vision.InteractiveSegmenterRegionOfInterest.Format.KEYPOINT, 
            keypoint=keypoint
        )
        rois.append(roi)
    
    # ROI 좌표를 NumPy 배열로 변환해서 시각화
    roi_coords = np.array([[roi.keypoint.x, roi.keypoint.y] for roi in rois])
    
    # 여러 개의 ROI를 scatter로 시각화
    plt.scatter(roi_coords[:, 0], roi_coords[:, 1])
    plt.title("Multiple ROIs Keypoints")
    plt.show()

    with vision.InteractiveSegmenter.create_from_options(segmenter_options) as segmenter:
        mask_np = np.zeros_like(mp_image.numpy_view()[:, :, 0], dtype=np.uint8)
        
        # 여러 ROI를 사용해서 segment 결과를 병합
        for roi in rois:
            mask_roi = segmenter.segment(mp_image, roi).category_mask.numpy_view().astype(np.uint8)
            mask_np = np.maximum(mask_np, mask_roi)  # 여러 마스크를 결합
        
        # 마스크 결과를 시각화 및 표시
        B = np.column_stack(np.where(mask_np > 128))
        B_sample = sample_points(B)

        plt.scatter(B[:, 0], B[:, 1])
        plt.title("Segmented Points from Multiple ROIs")
        plt.show()

        # 마스크 시각화
        cv2.imshow('Segmented Mask', mask_np * 255)
        #cv2.waitKey(0)

        if is_b_within_a(A, B_sample):
            return True
        else:
            return False

def create_binary_mask(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def pose_mode(image_path, criteria_image_path):
    original_image = cv2.imread(image_path)
    criteria_image = cv2.imread(criteria_image_path)

    if original_image is None or criteria_image is None:
        raise ValueError("One or both images not found.")


    resized_original_image, resized_criteria_image = resize_to_same_resolution(original_image, criteria_image)

    A = create_binary_mask(resized_criteria_image)

    extractor = PoseExtractor()

    landmarks = extractor.extract_pose_landmarks(resized_original_image)

    base_options = python.BaseOptions(model_asset_path='/Users/yangjiung/Documents/programming/brain power/magic_touch.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

    if landmarks.size > 0:
        return process_image(A, landmarks, options, image_path)
    else:
        print("랜드마크를 추출할 수 없습니다.")

#image_path = "capimgs/capture.png"
#criteria_image_path = "capimgs/criteria2.png"
