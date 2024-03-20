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

PanelDetectorHsv::PanelDetectorHsv(
    const int hue_min, const int hue_max, const int sat_min,
    const int sat_max, const int val_min, const int val_max)
    : hue_min(hue_min), hue_max(hue_max), sat_min(sat_min), sat_max(sat_max), val_min(val_min), val_max(val_max) {}

cv::Mat PanelDetectorHsv::hsv_filter(const cv::Mat& img) {
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

std::vector<Bbox> PanelDetectorHsv::detect(const cv::Mat& img) {
    cv::Mat img_filtered = hsv_filter(img);
    cv::Mat img_gray;
    cv::cvtColor(img_filtered, img_gray, cv::COLOR_BGR2GRAY);
    cv::Mat img_thresh;
    cv::threshold(img_gray, img_thresh, 100, 255, 0);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img_thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Bbox> bbox_list = create_bboxes(contours);
    std::vector<Bbox> merge_contours;
    while (bbox_list.size() > 0) {
        Bbox result(0, 0, 0, 0);
        auto rm_position = find_nearby_contours(bbox_list, 10, result);
        if (rm_position > 0) {
            merge_contours.push_back(result);
            bbox_list.erase(bbox_list.begin() + rm_position);
        } else {
            bbox_list.erase(bbox_list.begin());
        }
    }
    std::cout << "Detected " << merge_contours.size() << " panels" << std::endl;
    return merge_contours;
}

int PanelDetectorHsv::find_nearby_contours(const std::vector<Bbox>& bbox_list, const int w_mergin, Bbox& box) {
    if (bbox_list.size() == 1)
        return 0;

    int i = 0;
    for (size_t j = 1; j < bbox_list.size(); ++j) {
        auto x_mergin = (bbox_list[i].w + bbox_list[j].w) * 2 / 3;
        if (std::abs(bbox_list[i].x - bbox_list[j].x) < x_mergin && std::abs(bbox_list[i].w - bbox_list[j].w) < w_mergin) {
            box.x = std::min(bbox_list[i].x, bbox_list[j].x);
            box.y = std::min(bbox_list[i].y, bbox_list[j].y);
            box.w = std::max(bbox_list[i].w, bbox_list[j].w);
            box.h = std::abs(bbox_list[i].y - bbox_list[j].y) + std::max(bbox_list[i].h, bbox_list[j].h);
            return j;
        }
    }
    return 0;
}

std::vector<Bbox> PanelDetectorHsv::create_bboxes(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<Bbox> bbox_list;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        bbox_list.emplace_back(rect.x, rect.y, rect.width, rect.height);
    }
    return bbox_list;
}
    
}  // namespace core1_panel_detector
