// Copyright 2024 StrayedCats.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "core1_panel_detector_plugin/panel_detector_hsv.hpp"

namespace detector2d_plugins
{

void PanelDetectorHsv::init(const detector2d_parameters::ParamListener & param_listener)
{
  params_ = param_listener.get_params();

  int red_sat_min = 40, red_sat_max = 255, red_val_min = 40, red_val_max = 255;
  int blue_sat_min = 40, blue_sat_max = 255, blue_val_min = 40, blue_val_max = 255;
  
  int32_t red_hue_min = 0;
  int32_t red_hue_max = 50;

  int32_t blue_hue_min = 80;
  int32_t blue_hue_max = 130;

  hsvs.push_back(std::make_shared<core1_panel_detector::PanelDetectorHsv>(red_hue_min, red_hue_max, red_sat_min, red_sat_max, red_val_min, red_val_max));
  hsvs.push_back(std::make_shared<core1_panel_detector::PanelDetectorHsv>(blue_hue_min, blue_hue_max, blue_sat_min, blue_sat_max, blue_val_min, blue_val_max));
}

Detection2DArray PanelDetectorHsv::detect(const cv::Mat & image)
{
  auto img = image.clone();
  auto red_objects = hsvs[0]->detect(img);
  auto blue_objects = hsvs[1]->detect(img);

  if (this->params_.debug) {
    auto detected_img = img.clone();
    for (auto obj : red_objects) {
      cv::rectangle(detected_img, cv::Point(obj.x, obj.y), cv::Point(obj.x + obj.w, obj.y + obj.h), cv::Scalar(0, 0, 255), 2);
    }
    for (auto obj : blue_objects) {
      cv::rectangle(detected_img, cv::Point(obj.x, obj.y), cv::Point(obj.x + obj.w, obj.y + obj.h), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("panel_detector_hsv", detected_img);
    auto key = cv::waitKey(1);
    if (key == 27) {
      rclcpp::shutdown();
    }
  }

  auto red_boxes = this->objects_to_detection2d_array(red_objects, "person");
  auto blue_boxes = this->objects_to_detection2d_array(blue_objects, "bicycle");
  
  red_boxes.detections.insert(red_boxes.detections.end(), blue_boxes.detections.begin(), blue_boxes.detections.end());
  return red_boxes;
}

Detection2DArray PanelDetectorHsv::objects_to_detection2d_array(
  const std::vector<core1_panel_detector::Bbox> & objects,
  const std::string label)
{
  Detection2DArray boxes;
  for (auto obj : objects) {
    vision_msgs::msg::Detection2D detection;

    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = label;
    hypothesis.hypothesis.score = 1.0;
    detection.results.push_back(hypothesis);

    detection.bbox.center.position.x = obj.x + obj.w / 2;
    detection.bbox.center.position.y = obj.y + obj.h / 2;

    detection.bbox.size_x = obj.w;
    detection.bbox.size_y = obj.h;

    boxes.detections.push_back(detection);
  }
  return boxes;
}

}// namespace detector2d_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(detector2d_plugins::PanelDetectorHsv, detector2d_base::Detector)
