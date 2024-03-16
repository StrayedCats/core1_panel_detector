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


namespace core1_panel_detector
{

PanelDetectorHsv::PanelDetectorHsv(int hue_min, int hue_max, int sat_min, int sat_max, int val_min, int val_max)
    : hue_min(hue_min), hue_max(hue_max), sat_min(sat_min), sat_max(sat_max), val_min(val_min), val_max(val_max) {}

cv::Mat PanelDetectorHsv::hsv_filter(cv::Mat& img) {
    cv::Mat img_hsv;
    cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
    cv::Scalar lower(hue_min, sat_min, val_min);
    cv::Scalar upper(hue_max, sat_max, val_max);
    cv::Mat mask;
    cv::inRange(img_hsv, lower, upper, mask);
    cv::Mat result;
    cv::bitwise_and(img, img, result, mask);
    return result;
}

std::vector<Bbox> PanelDetectorHsv::detect(cv::Mat& img) {
    cv::Mat img_filtered = hsv_filter(img);
    cv::Mat img_gray;
    cv::cvtColor(img_filtered, img_gray, cv::COLOR_BGR2GRAY);
    cv::Mat img_thresh;
    cv::threshold(img_gray, img_thresh, 127, 255, 0);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img_thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Bbox> bbox_list = create_bboxes(contours);
    std::vector<Bbox> merge_contours;
    while (bbox_list.size() > 0) {
        Bbox result = find_nearby_contours(bbox_list, 10, 10);
        merge_contours.push_back(result);
    }
    return merge_contours;
}

Bbox PanelDetectorHsv::find_nearby_contours(std::vector<Bbox>& bbox_list, int x_mergin, int y_mergin) {
    Bbox box(0, 0, 0, 0);

    if (bbox_list.size() == 1)
        return bbox_list[0];

    for (size_t i = 0; i < bbox_list.size(); ++i) {
        for (size_t j = i + 1; j < bbox_list.size(); ++j) {
            if (std::abs(bbox_list[i].x - bbox_list[j].x) < x_mergin && std::abs(bbox_list[i].w - bbox_list[j].w) < y_mergin) {
                box.x = std::min(bbox_list[i].x, bbox_list[j].x);
                box.y = std::min(bbox_list[i].y, bbox_list[j].y);
                box.w = std::max(bbox_list[i].w, bbox_list[j].w);
                box.h = std::abs(bbox_list[i].y - bbox_list[j].y) + std::max(bbox_list[i].h, bbox_list[j].h);
                bbox_list.erase(bbox_list.begin() + i);
                bbox_list.erase(bbox_list.begin() + j - 1);
                return box;
            }
        }
        bbox_list.erase(bbox_list.begin() + i - 1);
        if (box.w != 0)
            break;
    }
    return box;
}

std::vector<Bbox> PanelDetectorHsv::create_bboxes(std::vector<std::vector<cv::Point>>& contours) {
    std::vector<Bbox> bbox_list;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        bbox_list.emplace_back(rect.x, rect.y, rect.width, rect.height);
    }
    return bbox_list;
}
    
}  // namespace core1_panel_detector
