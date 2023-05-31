#pragma once

#include <vector>
#include <color-helper.hh>
#include <texture-helper.hh>

namespace segmentation_helper
{
    using layer_vector = std::vector<uint8_t>;

    /**
     * Segment a frame into foreground and background
     * @param color_similarities The color similarities between the current frame and the background frame
     * @param texture_similarities The texture similarities between the current frame and the background frame
     * @return The a vector of boolean values (1 or 0) representing the foreground or the background
     */
    layer_vector segment(color_helper::similarity_vectors color_similarities, texture_helper::similarity_vector texture_similarities)
    {
        layer_vector segments;

        // Define the factors for the color and texture similarities
        std::vector<double> factors = {0.1, 0.3, 0.6};

        // Calculate the weighted sum of the color and texture similarities
        for (int i = 0; i < color_similarities[0].size(); i++)
        {
            double r = color_similarities[0][i];
            double g = color_similarities[1][i];
            double t = texture_similarities[i];

            // Sort the similarities in ascending order
            std::vector<double> similarities = {r, g, t};
            std::sort(similarities.begin(), similarities.end());

            // Multiply the similarities with the factors
            for (int j = 0; j < similarities.size(); j++)
            {
                similarities[j] *= factors[j];
            }

            // Calculate the weighted sum of the similarities and threshold it
            double similarity = similarities[0] + similarities[1] + similarities[2];

            // If the similarity is greater than 0.67, it is the foreground
            if (similarity >= 0.67)
            {
                segments.push_back(1);
            }
            // Otherwise, it is the background
            else
            {
                segments.push_back(0);
            }
        }

        return segments;
    }

} // namespace segmentation_helper
