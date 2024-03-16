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

#include "core1_panel_detector/panel_detector_hsv.hpp"

int main(int argc, char** argv)
{
    if(argc < 3){
        std::cerr << "Usage: " << argv[0] << " <color-str (red,blue..)> <path_to_image>" << std::endl;
        return -1;
    }

    std::string color_str = argv[1];
    int hue_min, hue_max, sat_min, sat_max, val_min, val_max;
    if (color_str == "red") {
        hue_min = 0;
        hue_max = 10;
        sat_min = 50;
        sat_max = 255;
        val_min = 50;
        val_max = 255;
    } else if (color_str == "blue") {
        hue_min = 80;
        hue_max = 150;
        sat_min = 50;
        sat_max = 255;
        val_min = 50;
        val_max = 255;
    } else {
        std::cerr << "Invalid color string" << std::endl;
        return -1;
    }

    core1_panel_detector::PanelDetectorHsv detector(hue_min, hue_max, sat_min, sat_max, val_min, val_max);
    std::string image_path = argv[2];
    std::string image_path_out = image_path.substr(0, image_path.find_last_of(".")) + "_out.jpg";

    cv::Mat frame = cv::imread(image_path);
    auto objects = detector.detect(frame);

    for (auto& obj : objects) {
        cv::rectangle(frame, cv::Point(obj.x, obj.y), cv::Point(obj.x + obj.w, obj.y + obj.h), cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(image_path_out, frame);

    return 0;
}