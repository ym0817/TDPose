#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <cmath>
#include <set>

using namespace std;
using namespace cv;


std::vector<std::vector<int>> BODY_PARTS_PAF_IDS = { {12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23}, {24, 25},
                                                       {0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {28, 29},
                                                       {30, 31}, {34, 35}, {32, 33}, {36, 37}, {18, 19}, {26, 27} };

vector<vector<int>> BODY_PART_KPT_IDS = { {1, 2}, { 1, 5 }, { 2, 3 }, { 3, 4 }, { 5, 6 },
    { 6, 7 }, { 1, 8 }, { 8, 9 }, { 9, 10 }, { 1, 11 },
    { 11, 12 }, { 12, 13 }, { 1, 0 }, { 0, 14 }, { 14, 16 },
    { 0, 15 }, { 15, 17 }, { 2, 16 }, { 5, 17 } };

cv::Mat normalize(const cv::Mat& img, const cv::Scalar& img_mean, float img_scale) {
    cv::Mat normalizedImg;
    img.convertTo(normalizedImg, CV_32F);

    normalizedImg -= img_mean;
    normalizedImg *= img_scale;

    return normalizedImg;
}

Mat padWidth(const cv::Mat& image, int stride, const cv::Scalar& pad_value, Point2i& min_dims, Vec4i& pad)
{
    int height = image.rows;
    int width = image.cols;

    height = min(min_dims.x, height);
    min_dims.x = ceil(min_dims.x / (float)stride) * stride;
    min_dims.y = max(min_dims.y, width);
    min_dims.y = ceil(min_dims.y / (float)stride) * stride;

    // Calculate the padding values
    pad[0] = (int)floor((min_dims.x - height) / 2.0);
    pad[1] = (int)floor((min_dims.y - width) / 2.0);
    pad[2] = (int)(min_dims.x - height - pad[0]);
    pad[3] = (int)(min_dims.y - width - pad[1]);

    // Apply padding using copyMakeBorder function
    cv::Mat padded_img;
    cv::copyMakeBorder(image, padded_img, pad[0], pad[2], pad[1], pad[3], cv::BORDER_CONSTANT, pad_value);

    return padded_img;

}

Mat input_preprocess(string img_path, float& scale, Vec4i& pad)
{
    Mat img = imread(img_path);

    int net_size = 256;
    scale = (float)net_size / (float)img.rows;
    Mat scaled_img;
    resize(img, scaled_img, Size(0, 0), scale, scale, INTER_LINEAR);


    Scalar img_mean(128, 128, 128);
    float img_scale = 1. / 255.;
    scaled_img = normalize(scaled_img, img_mean, img_scale);

    Point2i min_dims{ net_size, max(scaled_img.cols, net_size) };
    int stride = 8;
    Scalar pad_value(0, 0, 0);

    Mat padded_img = padWidth(scaled_img, stride, pad_value, min_dims, pad);
    cout << "padded img = " << padded_img.size << endl;

    return padded_img;
}

dnn::Net readmodel(string model_path)
{
    auto net = dnn::readNet(model_path);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    return net;
}

int extract_keypoints(cv::Mat& heatmap, std::vector<std::vector<cv::Vec4f>>& all_keypoints, int total_keypoint_num) {
    const float threshold = 0.1;

    // Apply threshold to heatmap
    Mat mask = heatmap < threshold;
    heatmap.setTo(0, mask);

    // Pad the heatmap with borders
    cv::copyMakeBorder(heatmap, heatmap, 2, 2, 2, 2, cv::BORDER_CONSTANT, 0);

    // Extract central heatmap and neighboring regions
    cv::Mat heatmap_center = heatmap(cv::Range(1, heatmap.rows - 1), cv::Range(1, heatmap.cols - 1));
    cv::Mat heatmap_left = heatmap(cv::Range(1, heatmap.rows - 1), cv::Range(2, heatmap.cols));
    cv::Mat heatmap_right = heatmap(cv::Range(1, heatmap.rows - 1), cv::Range(0, heatmap.cols - 2));
    cv::Mat heatmap_up = heatmap(cv::Range(2, heatmap.rows), cv::Range(1, heatmap.cols - 1));
    cv::Mat heatmap_down = heatmap(cv::Range(0, heatmap.rows - 2), cv::Range(1, heatmap.cols - 1));

    // Find heatmap peaks
    cv::Mat heatmap_peaks = (heatmap_center > heatmap_left) &
        (heatmap_center > heatmap_right) &
        (heatmap_center > heatmap_up) &
        (heatmap_center > heatmap_down);
    heatmap_peaks = heatmap_peaks(cv::Range(1, heatmap_center.rows - 1), cv::Range(1, heatmap_center.cols - 1));

    // Find coordinates of heatmap peaks
    std::vector<cv::Point> keypoints;
    cv::findNonZero(heatmap_peaks, keypoints);  // (w, h)

    // Sort keypoints based on the x-coordinate
    std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
        });

    std::vector<cv::Vec4f> keypoints_with_score_and_id;
    int keypoint_num = 0;
    std::vector<int> suppressed(keypoints.size(), 0);
    // cout << "keypoints num = " << keypoints.size() << endl;
    // Apply suppression to neighboring keypoints
    for (int i = 0; i < keypoints.size(); i++) {
        if (suppressed[i])
            continue;

        for (int j = i + 1; j < keypoints.size(); j++) {
            float distance = std::sqrt(std::pow(keypoints[i].x - keypoints[j].x, 2) +
                std::pow(keypoints[i].y - keypoints[j].y, 2));
            if (distance < 6)
                suppressed[j] = 1;
        }

        cv::Vec4f keypoint_with_score_and_id(keypoints[i].x, keypoints[i].y,
            heatmap.at<float>(keypoints[i]), total_keypoint_num + keypoint_num);
        keypoints_with_score_and_id.push_back(keypoint_with_score_and_id);
        keypoint_num++;
        // cout << keypoint_with_score_and_id << endl;
    }
    all_keypoints.push_back(keypoints_with_score_and_id);

    return keypoint_num;
}

void connection_nms(vector<int>& a_idx, vector<int>& b_idx, vector<float>& affinity_scores)
{
    vector<int> order(affinity_scores.size());

    for (int i = 0; i < order.size(); i++) {
        order[i] = i;
    }

    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int i, int j) {
        return affinity_scores[i] > affinity_scores[j];
        });

    vector<float> sorted_scores;
    vector<int> idx, sorted_a_idx, sorted_b_idx;
    set<int> has_kpt_a, has_kpt_b;

    for (int t : order) {
        int i = a_idx[t];
        int j = b_idx[t];
        if (has_kpt_a.find(i) == has_kpt_a.end() && has_kpt_b.find(j) == has_kpt_b.end()) {
            idx.push_back(t);
            sorted_a_idx.push_back(i);
            sorted_b_idx.push_back(j);
            sorted_scores.push_back(affinity_scores[t]);
            has_kpt_a.insert(i);
            has_kpt_b.insert(j);
        }
    }

    a_idx = sorted_a_idx;
    b_idx = sorted_b_idx;
    affinity_scores = sorted_scores;

}

void group_keypoints(Mat pafs, vector<vector<Vec4f>>& all_keypoints_by_type, vector<vector<float>>& pose_entries, vector<Vec4f>& all_keypoints)
{
    int points_per_limb = 10;
    int pose_entry_size = 20;
    std::vector<float> grid(points_per_limb);
    std::iota(grid.begin(), grid.end(), 0.0f);
    cout << "calling groupd keypoints" << endl;
    vector<Mat> pafs_split;
    split(pafs, pafs_split);

    for (auto n1 : all_keypoints_by_type) {
        for (auto n2 : n1) {
            all_keypoints.push_back(n2);
        }
    }

    for (int part_id = 0; part_id < BODY_PARTS_PAF_IDS.size(); part_id++) {

        // Get vectors between all pairs of keypoints, i.e.candidate limb vectors.
        vector<Mat> selected_part{ pafs_split[BODY_PARTS_PAF_IDS[part_id][0]], pafs_split[BODY_PARTS_PAF_IDS[part_id][1]] };
        Mat part_pafs;
        merge(selected_part, part_pafs);

        vector<Vec4f> kpts_a = all_keypoints_by_type[BODY_PART_KPT_IDS[part_id][0]];
        vector<Vec4f> kpts_b = all_keypoints_by_type[BODY_PART_KPT_IDS[part_id][1]];

        int n = kpts_a.size();
        int m = kpts_b.size();

        if (n * m == 0)
            continue;

        vector<Point2f> a;
        for (const auto& vec : kpts_a) {
            a.emplace_back(vec[0], vec[1]);
        }

        vector<Point2f> b;
        for (const auto& vec : kpts_b) {
            b.emplace_back(vec[0], vec[1]);
        }

        vector<Point2f> steps;
        vector<Point2f> vec_raw;
        float samples = 1. / ((float)points_per_limb - 1);
        for (int i = 0; i < b.size(); i++) {
            for (int j = 0; j < n; j++) {
                vec_raw.push_back(b[i] - a[j]);
                Point2f s = (b[i] - a[j]) * samples;
                steps.push_back(s);
            }
        }

        vector<int> x, y;
        for (int i = 0; i < steps.size(); i++) {
            for (int j = 0; j < points_per_limb; j++) {
                Point2i point = Point2i(steps[i] * j + a[i % n]);
                x.push_back(point.x);
                y.push_back(point.y);
            }
        }

        vector<Point2f> field; // size will be 120
        for (int i = 0; i < x.size(); i++) {
            // cout << part_pafs.at<Point2f>(y[i], x[i]) << endl;
            field.push_back(part_pafs.at<Point2f>(y[i], x[i]));
        }

        vector<Point2f> vec; // size will be 12
        for (const auto& v : vec_raw) {
            vec.push_back(v / (sqrt(v.x * v.x + v.y * v.y) + 1e-6));
        }

        float min_paf_score = 0.05;

        vector<vector<float>> affinity_scores;
        vector<vector<int>> valid_affinity_scores;
        vector<float> sums, success_ratios, affinity_scores2, affinity_scores3;
        vector<int> valid_nums, valid_scores, valid_limbs, a_idx, b_idx;
        for (int i = 0; i < field.size(); i++) {
            Point2f limb(field[i].x * vec[i / points_per_limb].x, field[i].y * vec[i / points_per_limb].y);
            // cout << limb << endl;
            sums.push_back(limb.x + limb.y);
            valid_scores.push_back((limb.x + limb.y) > min_paf_score);

            if ((i + 1) % 10 == 0) {
                affinity_scores.push_back(sums);
                valid_affinity_scores.push_back(valid_scores);
                valid_scores.clear();
                sums.clear();
            }
        }

        for (const auto& val : valid_affinity_scores) {
            valid_nums.push_back(accumulate(val.begin(), val.end(), 0));
        }

        for (auto valid_num : valid_nums) {
            success_ratios.push_back((float)valid_num / points_per_limb);
        }

        for (int i = 0; i < affinity_scores.size(); i++) {
            float sum = 0;
            for (int j = 0; j < affinity_scores[0].size(); j++) {
                sum += affinity_scores[i][j] * valid_affinity_scores[i][j];
            }
            affinity_scores2.push_back(sum / (valid_nums[i] + 1e-6));
        }

        for (int i = 0; i < affinity_scores2.size(); i++) {
             if (affinity_scores2[i] > 0 && success_ratios[i] > 0.8)
             {
                valid_limbs.push_back(i);
             }
        }

        if (valid_limbs.size() == 0)
            continue;

        for (int i = 0; i < valid_limbs.size(); i++) {
            b_idx.push_back(valid_limbs[i] / n);
            a_idx.push_back(valid_limbs[i] % n);
        }

        for (int i = 0; i < valid_limbs.size(); i++) {
            affinity_scores3.push_back(affinity_scores2[valid_limbs[i]]);
        }

        connection_nms(a_idx, b_idx, affinity_scores3);

        vector<Vec3f> connections;
        for (int i = 0; i < a_idx.size(); i++) {
            Vec3f connection = { kpts_a[a_idx[i]][3], kpts_b[b_idx[i]][3], affinity_scores3[i] };
            connections.push_back(connection);
        }

        if (connections.size() == 0)
            continue;

        if (part_id == 0) {
            vector<vector<float>> tmp_pose_entries(connections.size(), vector<float>(pose_entry_size, -1));
            pose_entries = tmp_pose_entries;

            for (int i = 0; i < connections.size(); i++) {
                pose_entries[i][BODY_PART_KPT_IDS[0][0]] = connections[i][0];
                pose_entries[i][BODY_PART_KPT_IDS[0][1]] = connections[i][1];
                pose_entries[i][pose_entry_size - 1] = 2;
                // pose_entries[i][pose_entry_size - 2] = all_keypoints_by_type[connections[i][]]
                vector<float> result;
                for (int j = 0; j < 2; j++) {
                    result.push_back(all_keypoints[connections[i][j]][2]);
                    // cout << all_keypoints[connections[i][j]][2] << " ";
                }
                // cout << endl;
                pose_entries[i][pose_entry_size - 2] = accumulate(result.begin(), result.end(), 0) + connections[i][2];
            }
        }

        else if (part_id == 17 || part_id == 18) {
            int kpt_a_id = BODY_PART_KPT_IDS[part_id][0];
            int kpt_b_id = BODY_PART_KPT_IDS[part_id][1];
            for (int i = 0; i < connections.size(); i++) {
                for (int j = 0; j < pose_entries.size(); j++) {
                    if (pose_entries[j][kpt_a_id] == connections[i][0] && pose_entries[j][kpt_b_id] == -1) {
                        pose_entries[j][kpt_b_id] = connections[i][1];
                    }
                    else if (pose_entries[j][kpt_b_id] == connections[i][1] && pose_entries[j][kpt_a_id] == -1) {
                        pose_entries[j][kpt_a_id] = connections[i][0];
                    }
                }
            }
            continue;
        }
        else {
            int kpt_a_id = BODY_PART_KPT_IDS[part_id][0];
            int kpt_b_id = BODY_PART_KPT_IDS[part_id][1];
            for (int i = 0; i < connections.size(); i++) {
                int num = 0;
                for (int j = 0; j < pose_entries.size(); j++) {
                    if (pose_entries[j][kpt_a_id] == connections[i][0]) {
                        pose_entries[j][kpt_b_id] = connections[i][1];
                        num += 1;
                        pose_entries[j][pose_entry_size - 1] += 1;
                        pose_entries[j][pose_entry_size - 2] += all_keypoints[connections[i][1]][2] + connections[i][2];
                    }
                }
                if (num == 0) {
                    vector<float> pose_entry(pose_entry_size, -1);
                    pose_entry[kpt_a_id] = connections[i][0];
                    pose_entry[kpt_b_id] = connections[i][1];
                    pose_entry[pose_entry_size - 1] = 2;
                    vector<float> result;
                    for (int j = 0; j < 2; j++) {
                        result.push_back(all_keypoints[connections[i][j]][2]);
                        // cout << all_keypoints[connections[i][j]][2] << " ";
                    }
                    pose_entry[pose_entry_size - 2] = accumulate(result.begin(), result.end(), 0) + connections[i][2];
                    pose_entries.push_back(pose_entry);
                }
            }
        }
    }

    vector<vector<float>> filtered_entries;
    for (int i = 0; i < pose_entries.size(); i++) {
        if ((pose_entries[i][pose_entry_size - 1] < 3) ||
            (pose_entries[i][pose_entry_size - 2] / pose_entries[i][pose_entry_size - 1] < 0.2)) {
            continue;
        }
        filtered_entries.push_back(pose_entries[i]);
    }
    pose_entries = filtered_entries;

}

// draw pose
void draw_pose(string img_path, vector<vector<float>> pose_entries, int num_keypoints, vector<Vec4f> all_keypoints, float scale, Vec4i pad)
{
    Mat img = imread(img_path);
    int upsample_ratio = 4;
    int stride = 8;

    // rescale keypoints
    for (int kpt_id = 0; kpt_id < all_keypoints.size(); kpt_id++) {
        all_keypoints[kpt_id][0] = (all_keypoints[kpt_id][0] * stride / upsample_ratio - pad[1]) / scale;
        all_keypoints[kpt_id][1] = (all_keypoints[kpt_id][1] * stride / upsample_ratio - pad[0]) / scale;
    }

    // cout << "num keypoints = " << num_keypoints << endl;
    for (int n = 0; n < pose_entries.size(); n++) 
    // for (int n = 0; n < 1; n++)
    {
        if (pose_entries[n].size() == 0)
            continue;
        vector<vector<int>> pose_keypoints(num_keypoints, vector<int>(2, -1));
        for (int kpt_id = 0; kpt_id < num_keypoints; kpt_id++) {
            if (pose_entries[n][kpt_id] != -1.0) // keypoint was found
            {
                pose_keypoints[kpt_id][0] = (int)(all_keypoints[(int)pose_entries[n][kpt_id]][0]);
                pose_keypoints[kpt_id][1] = (int)(all_keypoints[(int)pose_entries[n][kpt_id]][1]);
            }
        }

        // draw
        for (int part_id = 0; part_id < BODY_PARTS_PAF_IDS.size() - 2; part_id++) {
            int kpt_a_id = BODY_PART_KPT_IDS[part_id][0];
            int global_kpt_a_id = pose_keypoints[kpt_a_id][0];
            int x_a, y_a, x_b, y_b;
            if (global_kpt_a_id != -1) {
                x_a = (int)pose_keypoints[kpt_a_id][0];
                y_a = (int)pose_keypoints[kpt_a_id][1];
                circle(img, Point(x_a, y_a), 3, (0, 0, 255), 1);
            }

            int kpt_b_id = BODY_PART_KPT_IDS[part_id][1];
            int global_kpt_b_id = pose_keypoints[kpt_b_id][0];
            if (global_kpt_b_id != -1) {
                x_b = pose_keypoints[kpt_b_id][0];
                y_b = pose_keypoints[kpt_b_id][1];
                circle(img, Point(x_b, y_b), 3, (0, 0, 255), 1);
            }

            if (global_kpt_a_id != -1 && global_kpt_b_id != -1) {
                line(img, Point(x_a, y_a), Point(x_b, y_b), (0, 0, 255), 2);
            }
        }
    }

    imshow("rst", img);
    waitKey(0);
    destroyAllWindows();
}

Mat data2mat(int img_scale, int channels, vector<Mat> outputs, int outputLayer)
{
    int net_size = 32;
    float* data = (float*)outputs[outputLayer].data;
    vector<Mat> mats;
    for (int i = 0; i < channels; i++) {
        Mat layer(net_size, img_scale, CV_32F, data);        
        mats.push_back(layer);
        data += net_size * img_scale;

    }
    Mat rst;
    merge(mats, rst);
    
   return rst;
}

void dev1()
{
    string img_path = "../demo/3.jpg";
    int upsample_ratio = 4;
    float scale;
    Vec4i pad(0, 0, 0, 0);
    Mat img = input_preprocess(img_path, scale, pad);

    string model_path = "/home/ymm/WorkSpace/Pose/human-pose-estimation/human-pose-estimation.onnx";
    // string model_path = "leaky_dilated3.onnx";
    auto net = readmodel(model_path);
    auto blob = dnn::blobFromImage(img);
    net.setInput(blob);
    vector<Mat> outputs;

    vector<cv::String> outputLayers = net.getUnconnectedOutLayersNames();
    for (const auto& layer : outputLayers)
        cout << layer << endl;

    net.forward(outputs, net.getUnconnectedOutLayersNames());

    const int heatmap_size = 19;
    const int pafs_size = 38;
    int img_scale = img.cols / 8;
    Mat heatmaps = data2mat(img_scale, heatmap_size, outputs, 0);
    cout << "heatmap size = " << heatmaps.size << " " << heatmaps.channels() << endl;
    resize(heatmaps, heatmaps, Size(0, 0), upsample_ratio, upsample_ratio, INTER_CUBIC);

    Mat pafs = data2mat(img_scale, pafs_size, outputs, 1);
    resize(pafs, pafs, Size(0, 0), upsample_ratio, upsample_ratio, INTER_CUBIC);
    //cout << "pafs = " << pafs.size << " x " << pafs.channels() << endl;

    // decode
    int total_keypoints_num = 0;
    int num_keypoints = 18;
    vector<Mat> heatmaps_kpts;
    vector<vector<Vec4f>> all_keypoints_by_type;
    split(heatmaps, heatmaps_kpts);    

    for (int i = 0; i < num_keypoints; i++) {
        total_keypoints_num += extract_keypoints(heatmaps_kpts[i], all_keypoints_by_type, total_keypoints_num);
    }

    vector<vector<float>> pose_entries;
    vector<Vec4f> all_keypoints;

    group_keypoints(pafs, all_keypoints_by_type, pose_entries, all_keypoints);

    draw_pose(img_path, pose_entries, num_keypoints, all_keypoints, scale, pad);
}

int main()
{
    dev1();
}
