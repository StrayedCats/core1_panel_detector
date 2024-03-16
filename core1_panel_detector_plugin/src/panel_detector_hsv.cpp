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

  std::string color_str = "blue"; // TODO: get from params
  this->label = "person"; // TODO: get from params
  int hue_min, hue_max;
  int sat_min = 50, sat_max = 255, val_min = 50, val_max = 255;
  if (color_str == "red") {
    hue_min = 0;
    hue_max = 10;
  } else if (color_str == "blue") {
    hue_min = 80;
    hue_max = 150;
  } else {
    std::cerr << "Invalid color string" << std::endl;
    return;
  }

  hsv = std::make_shared<core1_panel_detector::PanelDetectorHsv>(hue_min, hue_max, sat_min, sat_max, val_min, val_max);
}

Detection2DArray PanelDetectorHsv::detect(const cv::Mat & image)
{
  auto img = image.clone();
  auto objects = hsv->detect(img);

  if (this->params_.yolox_trt_plugin.imshow_isshow) {
    auto detected_img = img.clone();
    for (auto obj : objects) {
      cv::rectangle(detected_img, cv::Point(obj.x, obj.y), cv::Point(obj.x + obj.w, obj.y + obj.h), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("panel_detector_hsv", detected_img);
    auto key = cv::waitKey(1);
    if (key == 27) {
      rclcpp::shutdown();
    }
  }

  auto boxes = this->objects_to_detection2d_array(objects);
  return boxes;
}

Detection2DArray PanelDetectorHsv::objects_to_detection2d_array(
  const std::vector<core1_panel_detector::Bbox> & objects)
{
  Detection2DArray boxes;
  for (auto obj : objects) {
    vision_msgs::msg::Detection2D detection;

    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = this->label;
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
