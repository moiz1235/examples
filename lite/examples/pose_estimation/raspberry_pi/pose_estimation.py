# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run pose classification and pose estimation."""
import argparse
import logging
import os
import sys
import time

import cv2
from ml import Classifier
from ml import Movenet
from ml import MoveNetMultiPose
from ml import Posenet
import numpy as np
from tflite_runtime.interpreter import Interpreter
import utils


def run(estimation_model: str, tracker_type: str, classification_model: str,
        label_file: str, camera_id: int, width: int, height: int, 
        input_video_file: str, save_video_out: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    estimation_model: Name of the TFLite pose estimation model.
    tracker_type: Type of Tracker('keypoint' or 'bounding_box').
    classification_model: Name of the TFLite pose classification model.
      (Optional)
    label_file: Path to the label file for the pose classification model. Class
      names are listed one name per line, in the same order as in the
      classification model output. See an example in the yoga_labels.txt file.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    input_video_file: The input_video_file to run pose_estimation on.
    save_video_out: Switch to save video output.
  """

  # Notify users that tracker is only enabled for MoveNet MultiPose model.
  if tracker_type and (estimation_model != 'movenet_multipose'):
    logging.warning(
        'No tracker will be used as tracker can only be enabled for '
        'MoveNet MultiPose model.')

  # Initialize the pose estimator selected.
  if estimation_model in ['movenet_lightning', 'movenet_thunder']:
    pose_detector = Movenet(estimation_model)
  elif estimation_model == 'posenet':
    pose_detector = Posenet(estimation_model)
  elif estimation_model == 'movenet_multipose':
    pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
  else:
    sys.exit('ERROR: Model is not supported.')

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  if input_video_file is not None:
    cap = cv2.VideoCapture(input_video_file)
    if (cap.isOpened() == False): 
      print("Error reading video file")
  else:
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  size = (frame_width, frame_height)

  if save_video_out:
    if input_video_file:
      video_split = os.path.split(os.path.abspath(input_video_file))
      out_vid_filename = os.path.join(*video_split[:-1], f"out_{video_split[-1]}")
    else:
      out_vid_filename = "output.mpeg"
    print(out_vid_filename)
    result = cv2.VideoWriter(out_vid_filename, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

  # Visualization parameters
  row_size = int(20 * frame_height/480)  # pixels
  left_margin = int(24 * frame_width/640)  # pixels
  text_color = (255, 0, 0)  # blue
  font_size = int(1 * frame_height/480)
  font_thickness = int(1 * frame_height/480)
  max_detection_results = 3
  fps_avg_frame_count = 10

  # Initialize the classification model
  if classification_model:
    interpreter = Interpreter(model_path=classification_model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    classifier = Classifier(classification_model, label_file)
    detection_results_to_show = min(max_detection_results,
                                    len(classifier.pose_class_names))

  # Continuously capture images from the camera and run inference
  try:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        if input_video_file is not None:
          print("Done processing input video")
          break
        else:
          sys.exit(
              'ERROR: Unable to read from webcam. Please verify your webcam settings.'
          )

      counter += 1
      if input_video_file is None:
        image = cv2.flip(image, 1)

      if estimation_model == 'movenet_multipose':
        # Run pose estimation using a MultiPose model.
        list_persons = pose_detector.detect(image)
      else:
        # Run pose estimation using a SinglePose model, and wrap the result in an
        # array.
        list_persons = [pose_detector.detect(image)]

      # Draw keypoints and edges on input image
      image = utils.visualize(image, list_persons)

      if classification_model:
        # Convert to float32 numpy array to match with the model's input data format.
        pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                                  for keypoint in list_persons[0].keypoints], dtype=np.float32)
        coordinates = pose_landmarks.flatten()
        coordinates = coordinates.reshape(1,-1)
        interpreter.set_tensor(input_index, coordinates)
        # Run inference.
        interpreter.invoke()
        # Find the class with highest probability.
        output = interpreter.tensor(output_index)
        predicted_label = np.argmax(output()[0])
        # print(predicted_label)

        # Run pose classification
        # prob_list = classifier.classify_pose(list_persons[0])

        # Show classification results on the image
        for i in range(detection_results_to_show):
          class_name = classifier.pose_class_names[i]
          probability = (1 if i==predicted_label else 0) * 1.0
          detect_text_color = tuple([255 if ((2-i)==j) else 0 for j in range(3)]) if i==predicted_label else text_color
          detect_text_color = text_color if detect_text_color==(0,0,0) else detect_text_color
          result_text = class_name + ' (' + str(probability) + ')'
          # print(result_text)
          text_location = (left_margin, (i + 2) * row_size)
          cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                      font_size, detect_text_color, font_thickness)

      # Calculate the FPS
      if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()

      # Show the FPS
      fps_text = 'FPS = ' + str(int(fps))
      text_location = (left_margin, row_size)
      cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                  font_size, text_color, font_thickness)

      if save_video_out:
        result.write(image)

      # Stop the program if the ESC key is pressed.
      k = cv2.waitKey(1) & 0xFF
      if k == 27:
        break

      if input_video_file is None:
        cv2.imshow(estimation_model, image)
  except:
    raise
  finally:
    print("Cleaning up!")
    cap.release()
    if input_video_file is not None and save_video_out:
      result.release()
    cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of estimation model.',
      required=False,
      default='movenet_lightning')
  parser.add_argument(
      '--tracker',
      help='Type of tracker to track poses across frames.',
      required=False,
      default='bounding_box')
  parser.add_argument(
      '--classifier', help='Name of classification model.', required=False)
  parser.add_argument(
      '--label_file',
      help='Label file for classification.',
      required=False,
      default='labels.txt')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  parser.add_argument(
      '--input_video_file',
      help='video to process instead of camera',
      required=False,
      default=None)
  parser.add_argument(
      '--save_video_out',
      help='output a video instead of display',
      default=False, 
      action='store_true')
  args = parser.parse_args()
  # args = parser.parse_args(['--model', 'movenet_thunder', '--classifier', 'pose_classifier.tflite', 
  #                           '--label_file', 'pose_labels.txt', '--input_video_file', 'pranshul_posture.mp4',
  #                           '--save_video_out'])
  run(args.model, args.tracker, args.classifier, args.label_file,
      int(args.cameraId), args.frameWidth, args.frameHeight, args.input_video_file,
      args.save_video_out)
  
if __name__ == '__main__':
  main()
