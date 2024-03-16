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

#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

namespace core1_panel_detector
{

class Bbox {
    public:
        int x, y, w, h;

        Bbox(int x, int y, int w, int h) : x(x), y(y), w(w), h(h) {}

        std::string to_string() const {
            return "x: " + std::to_string(x) + ", y: " + std::to_string(y) +
                ", w: " + std::to_string(w) + ", h: " + std::to_string(h);
        }
};

class PanelDetectorHsv {
    private:
        int hue_min, hue_max, sat_min, sat_max, val_min, val_max;

    public:
        PanelDetectorHsv(const int, const int, const int,
                        const int, const int, const int);

        cv::Mat hsv_filter(const cv::Mat &);
        std::vector<Bbox> detect(const cv::Mat &);

        int find_nearby_contours(const std::vector<Bbox> &, const int, const int, Bbox &);
        std::vector<Bbox> create_bboxes(const std::vector<std::vector<cv::Point>> &);
};

}  // namespace core1_panel_detector

