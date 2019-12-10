using System;
using System.Drawing;
using System.IO;
using Emgu.CV; 
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;


namespace EmguTF_pose
{
    /// <summary>
    /// A posenet estimator is a deep neural network loaded with Emgu.TF.Lite. It 
    /// generates a 3D tensor of heatmaps and a 3D tensor of offsets when fed with an 
    /// image. The idea is to estimate the keypoint position on a regular grid, then to
    /// translate it to its real location in the input image dimension with an offset vector.
    /// We will neglect the other outputs of the network for now (they are useful for multi-body 
    /// pose estimation only).
    /// </summary>
    class PoseNetEstimator
    {
        /// <summary>
        /// A path to a frozen deep neural network saved with tensorflow lite.
        /// </summary>
        private String m_frozenModelPath = "";

        /// <summary>
        /// Expected extension for a frozen model (.tflite)
        /// </summary>
        private String m_expectedModelExtension = ".tflite";

        /// <summary>
        /// Our model abstraction.
        /// </summary>
        private Emgu.TF.Lite.FlatBufferModel m_model = null;

        /// <summary>
        /// An interpreter for our model.
        /// </summary>
        private Emgu.TF.Lite.Interpreter m_interpreter = null;

        /// <summary>
        /// Our input tensor standing for an input RGB image.
        /// </summary>
        private Emgu.TF.Lite.Tensor m_inputTensor = null;

        /// <summary>
        /// An array of output tensors.
        /// * Index 0. The heatmap tensor It is a 3D tensor of size resolution x resolution x 17 (number of keypoints)
        ///            where each channel represents the probability of a specified keypoint
        ///            on a regular grid (size resolution). The number 17 is known a-priori with PoseNet.
        ///             + heatmaps.Dims[0] : batch size = 1
        ///             + heatmaps.Dims[1] : resolution = W
        ///             + heatmaps.Dims[2] : resolution = H
        ///             + heatmaps.Dims[3] : channels (1/keypoint) = 17 with PoseNet
        /// * Index 1. The offset tensor. It is a 3D tensor of size resolution x resolution x 34 (twice more channels)
        ///            where channels 0-16 are X axis offset, and channels 17-33 are Y axis offsets
        ///             + offsets.Dims[0] : batch size = 1
        ///             + offsets.Dims[1] : resolution = W 
        ///             + offsets.Dims[2] : resolution = H 
        ///             + offsets.Dims[3] : channels (2/keypoint; 1 for X, 1 for Y) = 34 with PoseNet
        /// * For both : resolution Resolution = ((InputImageSize - 1) / OutputStride) + 1
        ///              where InputImageSize is input dependent. 
        ///              Similarly: outputstride = (inputImage size - 1) / (Resolution -1)
        /// </summary>
        private Emgu.TF.Lite.Tensor[] m_outputTensors = null;

        /// <summary>
        /// A vector storing ordered body joint keypoint heatmaps as Emgu.CV.Mat objects.
        /// </summary>
        private VectorOfMat m_heatmapsChannels = new VectorOfMat(m_numberOfKeypoints);

        /// <summary>
        /// A vector storing the generated and ordered body joint keypoint offsets as Emgu.CV.Mat objects.
        /// </summary>
        private VectorOfMat m_offsetsChannels  = new VectorOfMat(m_numberOfKeypoints);

        /// <summary>
        /// The number of keypoints we can find. This is an a-priori knowledge based on the network architecture.
        /// We have 17 keypoints per body to find with PoseNet. Their names are stored in <see cref="m_keypointsNames"/>, while the 
        /// keypoints themselve are updated after each forward pass/inference in <see cref="m_keypoints"/>.
        /// </summary>
        private const int m_numberOfKeypoints = 17;

        /// <summary>
        /// An array of <see cref="Point"/> representing the keypoints found with posenet on an input image.
        /// Each keypoint is obtained as the maximum location from the estimated heatmaps stored in <see cref="m_heatmapsChannels"/>.
        /// The number of keypoints we retrieve is given by <see cref="m_numberOfKeypoints"/>.
        /// The name for each keypoint is stored in <see cref="m_keypointNames"/>.
        /// </summary>
        public Point[] m_keypoints = new Point[m_numberOfKeypoints];

        /// <summary>
        /// An array of <see cref="string"/> storing the name of the ordered <see cref="m_keypoints"/> found with posenet.
        /// </summary>
        private string[] m_keypointName = new string[m_numberOfKeypoints]{
                    "nose", "left eye", "right eye", "left ear", "right ear", "left shoulder",
                    "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
                    "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"
        };

        /// <summary>
        /// Default constructor. It does nothing but allocating memory. Use the constructor with arguments.
        /// </summary>
        public PoseNetEstimator()
        {
        }

        /// <summary>
        /// Constructor with arguments defining the values of <see cref="m_frozenModelPath"/>, <see cref="m_model"/>,
        /// <see cref="m_interpreter"/>. It also defines <see cref="m_outputTensors"/> as the <see cref="m_interpreter.Outputs">
        /// and <see cref="m_inputTensor"/> as <see cref="m_inputTensor.Inputs[0]"/>, assuming the input tensor will be a 
        /// 3 channels BGR image.
        /// </summary>
        /// <param name="frozenModelPath">Path to a PoseNet model saved with tensorflow lite (.tflite file).</param>
        /// <param name="numberOfThreads">Number of threads the neural network will be able to use (default: 2, from base class)</param>
        public PoseNetEstimator(String frozenModelPath,
                                int numberOfThreads = 4)
        {
            // Check file
            if (!File.Exists(frozenModelPath))
            {
                Console.WriteLine("ERROR:");
                Console.WriteLine("FrozenModelPath specified in DeepNetworkLite " +
                                  "construtor with argument does not exist.");
                Console.WriteLine("Network not loaded.");
                return;
            }
            if (Path.GetExtension(frozenModelPath) != m_expectedModelExtension)
            {
                Console.WriteLine("ERROR:");
                Console.WriteLine("Extension of specified frozen model path in DeepNetworkLite " +
                                  "constructor with argument does not" +
                                  "match " + m_expectedModelExtension);
                Console.WriteLine("Network not loaded.");
                return;
            }

            if (m_frozenModelPath == "")
                m_frozenModelPath = frozenModelPath;

            try
            {
                if (m_frozenModelPath != "")
                {
                    m_model = new Emgu.TF.Lite.FlatBufferModel(filename: m_frozenModelPath);
                    m_interpreter = new Emgu.TF.Lite.Interpreter(flatBufferModel: m_model);
                    m_interpreter.AllocateTensors();
                    m_interpreter.SetNumThreads(numThreads: numberOfThreads);
                }
            }
            catch
            {
                DisposeObjects();

                Console.WriteLine("ERROR:");
                Console.WriteLine("Unable to load frozen model in DeepNetworkLite constructor with arguments " +
                                  "despite files was found with correct extension. " +
                                  "Please, make sure you saved your model using tensorflow lite pipelines." +
                                  "Current path found is : " + m_frozenModelPath);
                return;
            }

            if (m_inputTensor == null)
            {

                int[] input = m_interpreter.InputIndices;
                m_inputTensor = m_interpreter.GetTensor(input[0]);

            }

            if (m_outputTensors == null)
            {
                m_outputTensors = m_interpreter.Outputs;
            }

            return;
        }

        /// <summary>
        /// Desctructor. Call <see cref="DisposeObjects"/>.
        /// </summary>
        ~PoseNetEstimator()
        {
            DisposeObjects();
        }

        /// <summary>
        /// Dispose our model and interpreter.
        /// </summary>
        public void DisposeObjects()
        {
            if (m_model != null)
            {
                m_model.Dispose();
                m_model = null;
            }

            if (m_interpreter != null)
            {
                m_interpreter.Dispose();
                m_interpreter = null;
            }
        }

        /// <summary>
        /// Apply sigmoid on a value
        /// </summary>
        /// <param name="value"></param>
        /// <returns> 1 / (1 / exp(-value))</returns>
        public double sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }


        /// <summary>
        /// Perform a forward pass on the image using the current <see cref="m_interpreter"/>.
        /// We assume a PoseNetEstimator instance constructed with the constructor with arguments.
        /// </summary>
        /// <param name="inputImage">A RGB image. It will be resized during the inference to match the network's input size.</param>
        /// <returns>
        /// On error : An empty array of Points (size 0 ; new Point[0]).
        /// On success : Return an array of 17 points <see cref="m_numberOfKeypoints"/> representing 17 human body keypoints. 
        ///              If the probability of a keypoint is too low (hardcoded threshold for now, see below),
        ///              keypoint is set to Point(-1,-1). The points are returned in the dimension
        ///              of the network's input size (e.g., 257x257). 
        ///              You may need to further interpolate them for display purpose. A useful formula is 
        ///              newX = (currentX / currentWidth) * newWidth (same for y , height). 
        ///          
        /// </returns>
        public Point[] Inference(Emgu.CV.Mat inputImage)
        {
            // Forward pass
            // Is the input empty ?
            if (inputImage.IsEmpty)
            {
                Console.WriteLine("ERROR:");
                Console.WriteLine("Empty image given to InferenceOnImage in DeepNetworkLite classe. " +
                                    "Return.");
                return new Point[0];
            }

            // Is the input encoded with 32 bit floating point precision ?
            if (inputImage.Depth != Emgu.CV.CvEnum.DepthType.Cv32F)
            {
                inputImage.ConvertTo(inputImage, Emgu.CV.CvEnum.DepthType.Cv32F);
                inputImage /= 255;
            }

            using (Mat image = inputImage)
            {
                try
                {
                    // Load image in interpreter using ReadTensorFromMatBgr function from the utils.
                    Utils.ReadTensorFromMatBgr(
                        image: image,
                        tensor: m_inputTensor,
                        inputHeight: m_inputTensor.Dims[1],
                        inputWidth: m_inputTensor.Dims[2]
                    );

                    // Actually perfom the inference
                    m_interpreter.Invoke();
                }
                catch
                {
                    Console.WriteLine("ERROR:");
                    Console.WriteLine("Unable to invoke interpreter in DeepNetworkLite.");
                    return new Point[0];
                }
            }

            // 1- Converts 3D tensors to Emgu.CV.Mat - 9 is the resolution here
            Emgu.CV.Mat heatmaps_mat = new Emgu.CV.Mat();
            Emgu.CV.Mat offsets_mat  = new Emgu.CV.Mat();
            try
            {
                heatmaps_mat = new Mat(m_outputTensors[0].Dims[1], m_outputTensors[0].Dims[2], 
                                       DepthType.Cv32F, m_numberOfKeypoints, m_outputTensors[0].DataPointer,
                                       sizeof(float) * 3 * m_outputTensors[0].Dims[1]);
                offsets_mat = new Mat(m_outputTensors[1].Dims[1], m_outputTensors[1].Dims[2],
                                       DepthType.Cv32F, m_numberOfKeypoints*2, m_outputTensors[1].DataPointer,
                                      sizeof(float) * 3 * m_outputTensors[1].Dims[1]);
            }
            catch
            {
                Console.WriteLine("Unable to read heatmaps or offsets in PoseNetEstimator. " +
                                  "Return new Point[0] - empty array of Points.");
                return new Point[0];
            }

            // 2 - Split channels and store them in vector of mat
            if (!heatmaps_mat.IsEmpty & !offsets_mat.IsEmpty)
            {
                Emgu.CV.CvInvoke.Split(heatmaps_mat, m_heatmapsChannels);
                Emgu.CV.CvInvoke.Split(offsets_mat,m_offsetsChannels);
            }
            else
            {
                return new Point[0];
            }

            // 3 - Get max prob on heatmap and apply offset :D
            try
            {
                for (var i = 0; i < m_numberOfKeypoints; i++) // 11 and not 17 to keep only upper body keypoints - todo: remove hardcoded
                {
                    var maxLoc = new Point();
                    var minLoc = new Point();
                    double min = 0;
                    double max = 0;

                    Emgu.CV.CvInvoke.MinMaxLoc(m_heatmapsChannels[i], ref min, ref max, ref minLoc, ref maxLoc);

                    if (sigmoid(max) > 0.05) // 0.05 is a fixed probability threshold between 0 and 1 - todo: remove hardcoded
                    {
                        Image<Gray, Single> offset_y = m_offsetsChannels[i].ToImage<Gray, Single>();
                        Image<Gray, Single> offset_x = m_offsetsChannels[i + 17].ToImage<Gray, Single>();
                        var y = offset_y[maxLoc.Y, maxLoc.X];
                        var x = offset_x[maxLoc.Y, maxLoc.X];

                        int output_stride = (this.m_interpreter.Inputs[0].Dims[1] - 1) / (this.m_interpreter.Outputs[0].Dims[1] - 1);
                        m_keypoints[i] = new Point((maxLoc.X * output_stride + (int)x.Intensity), 
                                                   (maxLoc.Y * output_stride + (int)y.Intensity));
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
            return m_keypoints;
        }
    }
}
