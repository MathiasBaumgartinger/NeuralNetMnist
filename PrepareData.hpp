//
// Created by mathias on 6/5/19.
//



#include <vector>
#include <cstdint>

namespace Data {
    std::vector<std::vector<double>> prepare(std::vector<std::vector<uint8_t>>& images) {

        auto normalized = std::vector<std::vector<double>>();

        for (int row_iterator = 0; row_iterator < images.size(); row_iterator++) {
            normalized.emplace_back(std::vector<double>());
            for (int column_iterator = 0; column_iterator < images[row_iterator].size(); column_iterator++) {
                auto pixel = images[row_iterator][column_iterator];

                double d_pixel;
                d_pixel = (double) 2* pixel / 255 - 1.0;

                normalized[row_iterator].push_back(d_pixel);
            }
        }

        return normalized;
    }

    std::vector<std::vector<double>> prepare_labels (std::vector<uint8_t> labels) {
        std::vector<std::vector<double>> prepared_labels;

        for (int label = 0; label < labels.size(); label++) {
            prepared_labels.emplace_back();
            for (int j = 0; j < 10; j++) {
                if (j == labels[label]) {
                    prepared_labels[label].push_back(1.0);
                } else {
                    prepared_labels[label].push_back(0.0);
                }
            }
        }

        return prepared_labels;
    }

}
