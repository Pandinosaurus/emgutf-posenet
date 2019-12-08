using System;
using System.Drawing;
using Emgu.CV; //VideoCapture, Mat
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguTF_pose
{
    /// <summary>
    /// A posenet estimator is a deep neural network loaded with Emgu.TF.Lite. It will
    /// find joint keypoints based on estimated joint heatmaps and offset vectors.
    /// </summary>
    class PoseNetEstimator : DeepNetworkLite 
    {
        /// <summary>
        /// The keypoints found with posenet.
        /// </summary>
        public Point[] m_keypoints;

        /// <summary>
        /// The name of the keypoints found with posenet.
        /// </summary>
        public string[] m_keypointNames;

        /// <summary>
        /// The number of keypoints we can find.
        /// This is an a-priori knowledge based on the network architecture.
        /// We have 17 keypoints per body to find with PoseNet.
        /// </summary>
        public const int m_numberOfKeypoints = 17;

        /// <summary>
        /// Default constructor. It does nothing but allocating memory. 
        /// You need to specify a frozen model path to make it works.
        /// TIP : use the constructor with arguments, this one is useless for now.
        /// </summary>
        public PoseNetEstimator() { }

        /// <summary>
        /// Constructor with arguments interfacing with the base constructor with argument.
        /// </summary>
        /// <param name="frozenModelPath">Path to a deep neural network frozen and saved with tensorflow lite.</param>
        /// <param name="numberOfThreads">Number of threads the neural network will be able to use (default: 2)</param>
        public PoseNetEstimator(String frozenModelPath,
                             int numberOfThreads) : base(frozenModelPath,
                                                         numberOfThreads)
        {
            m_keypoints = new Point[m_numberOfKeypoints];
            m_keypointNames = new string[m_numberOfKeypoints]{
                    "nose", "left eye", "right eye", "left ear", "right ear", "left shoulder",
                    "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
                    "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"};
        }

        /// <summary>
        /// Perform a forward pass on the image using the current PoseNetEstimator.
        /// We assume the PoseNetEstimator was constructed with the constructor with arguments.
        /// </summary>
        /// <param name="image">A RGB image. It will be resized during the inference to match the network's input size.</param>
        /// <returns>An array of 17 points representing 17 human body keypoints. If the probability of a keypoint is too low
        ///          (hardcoded threshold for now, see below), it is set to Point(-1,-1). The points are returned in the dimension
        ///          of the network's input size (e.g. 257x257). You may need to interpolate them for display purpose. A useful
        ///          formula is newX = (currentX / currentWidth) * newWidth (same for y , height). Additionaly, you may want to check
        ///          the list of the ordered keypoints. In this function, we enfore lower body part to not be found (i.e., we set the
        ///          points to Point(-1,-1) for all keypoints 12 to 17.
        ///          
        ///          Ordered keypoints list :
        ///          
        /// </returns>
        public Point[] Inference(Emgu.CV.Mat image)
        {
            // Forward pass
            Emgu.TF.Lite.Tensor[] inference_output = InferenceOnImage(image);

            // Check output names and retrive the data.
            // foreach (var output in interpreter.Outputs)
            // {
            //      Console.WriteLine(inference_output.Name);
            // }
            // ==> heatmaps, offsets, other...
            //
            // * The heatmap is a 3D tensor of size resolution x resolution x 17 (number of keypoints)
            //   where each channel represents the probability of a specified keypoint
            //   on a regular grid (size resolution)
            // * The offset is a 3D tensor of size resolution x resolution x 34 (twice more channels)
            //   where channels 0-16 are X axis offset, and channels 17-33 are Y axis offsets
            // * For both resolution Resolution = ((InputImageSize - 1) / OutputStride) + 1
            //   Example: an input image with a width of 225 pixels and an output
            //            stride of 16 results in an output resolution of 15
            //            15 = ((225 - 1) / 16) + 1
            //   Other example : In case we have InputImageSize == 257 and OutputStide == 32, 
            //                   we get an output resolution of 9 = ((257-1)/32 +1)
            //
            // -----------------------------------------------------------------------
            // Shapes:
            //    inference_output[0] : heatmaps
            //         heatmaps.Dims[0] : batch size = 1
            //         heatmaps.Dims[1] : resolution = W
            //         heatmaps.Dims[2] : resolution = H
            //         heatmaps.Dims[3] : chennels (1/keypoint) = 17 with PoseNet
            //    inference_output[1] : offsets
            //         offsets.Dims[0] : batch size = 1
            //         offsets.Dims[1] : resolution = W 
            //         offsets.Dims[2] : resolution = H 
            //         offsets.Dims[3] : chennels (2/keypoint; 1 for X, 1 for Y) = 32 with PoseNet
            // ------------------------------------------------------------------------
            // 1- Converts to Emgu.CV.Mat - 9 is the resolution here
            Emgu.CV.Mat heatmaps_mat = new Emgu.CV.Mat();
            Emgu.CV.Mat offsets_mat = new Emgu.CV.Mat();
            try
            {
                heatmaps_mat = new Mat(9, 9, DepthType.Cv32F, 17, inference_output[0].DataPointer,
                                       sizeof(float) * 3 * inference_output[0].Dims[1]);
                offsets_mat = new Mat(9, 9, DepthType.Cv32F, 34, inference_output[1].DataPointer,
                                      sizeof(float) * 3 * inference_output[1].Dims[1]);
            }
            catch
            {
                Console.WriteLine("Unable to read heatmaps or offsets in PoseNetEstimator. " +
                                  "Return new Point[0] - empty array of Points.");
                return new Point[0];
            }

            // 2 - Split channels
            var heatmaps_channels = new VectorOfMat();
            var offsets_channels = new VectorOfMat();
            if (!heatmaps_mat.IsEmpty & !offsets_mat.IsEmpty)
            {
                Emgu.CV.CvInvoke.Split(heatmaps_mat, heatmaps_channels);
                Emgu.CV.CvInvoke.Split(offsets_mat, offsets_channels);
            }
            else
            {
                return new Point[0];
            }

            // 3 - Get max prob on heatmap and apply offset :D
            try
            {
                for (var i = 0; i < 11; i++) // 11 and not 17 to keep only upper body keypoints - todo: remove hardcoded
                {
                    var maxLoc = new Point();
                    var minLoc = new Point();
                    double min = 0;
                    double max = 0;

                    Emgu.CV.CvInvoke.MinMaxLoc(heatmaps_channels[i], ref min, ref max, ref minLoc, ref maxLoc);

                    if (sigmoid(max) > 0.05) // 0.05 is a fixed probability threshold between 0 and 1 - todo: remove hardcoded
                    {
                        Image<Gray, Single> offset_y = offsets_channels[i].ToImage<Gray, Single>();
                        Image<Gray, Single> offset_x = offsets_channels[i + 17].ToImage<Gray, Single>();
                        var y = offset_y[maxLoc.Y, maxLoc.X];
                        var x = offset_x[maxLoc.Y, maxLoc.X];

                        m_keypoints[i] = new Point((maxLoc.X * 32 + (int)x.Intensity), (maxLoc.Y * 32 + (int)y.Intensity));
                        // 32 is the output stride
                    }
                    else
                    {
                        m_keypoints[i] = new Point(-1, -1);
                    }
                }
            }
            catch
            {
                Console.WriteLine("Error in PoseNetEstimator Inference : unable to decode heatmaps and offsets. " +
                                  "Return new Point[0] - empty array of points.");
                return new Point[0];
            }

            // Dispose
            heatmaps_mat.Dispose();
            offsets_mat.Dispose();
            for(var i=0; i<17; i++)
            {
                heatmaps_channels[i].Dispose();
                offsets_channels[i].Dispose();
            }
            return m_keypoints;
        }
    }
}
