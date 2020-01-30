package com.example.opencvapp;

import org.opencv.calib3d.StereoSGBM;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class DepthDetector {
    private Mat createDisparityMap(Mat rectLeft, Mat rectRight){

        // Converts the images to a proper type for stereoMatching
        Mat left = new Mat();
        Mat right = new Mat();

        Imgproc.cvtColor(rectLeft, left, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(rectRight, right, Imgproc.COLOR_BGR2GRAY);

        // Create a new image using the size and type of the left image
        Mat disparity = new Mat(left.size(), left.type());

        int numDisparity = (int)(left.size().width/8);

        StereoSGBM stereoAlgo = StereoSGBM.create(
                0,    // min DIsparities
                numDisparity, // numDisparities
                11,   // SADWindowSize
                2*11*11,   // 8*number_of_image_channels*SADWindowSize*SADWindowSize   // p1
                5*11*11,  // 8*number_of_image_channels*SADWindowSize*SADWindowSize  // p2

                -1,   // disp12MaxDiff
                63,   // prefilterCap
                10,   // uniqueness ratio
                0, // sreckleWindowSize
                32, // spreckle Range
                0); // full DP
        // create the DisparityMap - SLOW: O(Width*height*numDisparity)
        stereoAlgo.compute(left, right, disparity);

        Core.normalize(disparity, disparity, 0, 256, Core.NORM_MINMAX);

        return disparity;
    }
}
