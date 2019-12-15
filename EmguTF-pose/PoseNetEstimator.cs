using System;
using System.Drawing;
using System.IO;
using Emgu.CV; 
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Collections.Generic;

namespace EmguTF_pose
{
    /// <summary>
    /// Enumerate available body parts. 
    /// 
    /// Note:
    /// Last enumeration value NUMBER_MAX is used to get the number of body parts. 
    /// It is useful to create fixed size arrays and vectors for body parts.
    /// We need it because the enum block start counting to 0 and retrieving the 
    /// number of elements in an enumeration is syntaxically complex.
    /// </summary>
    enum BodyParts
    {
        NOSE,
        LEFT_EYE,
        RIGHT_EYE,
        LEFT_EAR,
        RIGHT_EAR,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE,
        NUMBER_MAX
    }

    /// <summary>
    /// A class representing a keypoint in posenet. Each keypoint will be related to a body part from <see cref="BodyParts"/>.
    /// It will be assigned a position and a floating point precision score. The score will be the probability of the keypoint
    /// being a body part.
    /// </summary>
    class Keypoint
    {
        public int bodyPart = -1;
        public Point position = new Point(-1, -1);
        public Point position_raw = new Point(-1, -1);
        public float score = 0.0f;

        public Keypoint() { }
        public void reset()
        {
            bodyPart = -1;
            position.X = -1;
            position.Y = -1;
            position_raw.X = -1;
            position_raw.Y = -1;
            score = 0.0f;
        }
    }

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
        /// * For all : resolution Resolution = ((InputImageSize - 1) / OutputStride) + 1
        ///              where InputImageSize is input dependent. 
        ///              Similarly: outputstride = (inputImage size - 1) / (Resolution -1)
        /// </summary>
        private Emgu.TF.Lite.Tensor[] m_outputTensors = null;

        /// <summary>
        /// A vector storing ordered body joint keypoint heatmaps as Emgu.CV.Mat objects.
        /// It is filled by splitting the heatmap retrieved from <see cref="m_outputTensors[0]"/>
        /// along the channels dimension.
        /// </summary>
        private VectorOfMat m_heatmapsChannels = new VectorOfMat(m_numberOfKeypoints);

        /// <summary>
        /// A vector storing the generated and ordered body joint keypoint offsets as Emgu.CV.Mat objects.
        /// First 17 channels are for the X axis offsets, last 17 channels are for the X axis offsets.
        /// It is filled by splitting the heatmap from <see cref="m_outputTensors[1]"/>
        /// along the channels dimension.
        /// </summary>
        private VectorOfMat m_offsetsChannels = new VectorOfMat(m_numberOfKeypoints*2);

        /// <summary>
        /// TODO
        /// </summary>
        private VectorOfMat m_forwardDisplacementChannels = new VectorOfMat((m_numberOfKeypoints-1)*2);

        /// <summary>
        /// TODO
        /// </summary>
        private VectorOfMat m_backwardDisplacementChannels = new VectorOfMat((m_numberOfKeypoints-1) * 2);

        /// <summary>
        /// The number of body part keypoints we can find. 
        /// This is an a-priori knowledge based on the network architecture.
        /// We have 17 keypoints per body to find with PoseNet. 
        /// The keypoint are updated after each forward pass/inference in <see cref="m_keypoints"/>.
        /// </summary>
        private const int m_numberOfKeypoints = (int)BodyParts.NUMBER_MAX;

        /// <summary>
        /// An array of <see cref="Point"/> representing the keypoints found with posenet on an input image.
        /// Each keypoint is obtained as the maximum location from the estimated heatmaps stored in <see cref="m_heatmapsChannels"/>.
        /// The number of keypoints we retrieve is given by <see cref="m_numberOfKeypoints"/>.
        /// The name for each keypoint is stored in <see cref="m_keypointNames"/>.
        /// </summary>
        public Keypoint[] m_keypoints = new Keypoint[m_numberOfKeypoints];

        /// <summary>
        /// Define the joints between keypoints. A joint is, basically, a line between two keypoints.
        /// It is used to draw the lines between the keypoints.
        /// We only define uper body joints here.
        /// </summary>
        public int[][] m_keypointsJoints = new int[][]
        {
            new int[2]{ (int)BodyParts.NOSE, (int)BodyParts.LEFT_EYE }, 
            new int[2]{ (int)BodyParts.NOSE, (int)BodyParts.RIGHT_EYE },
            new int[2]{ (int)BodyParts.LEFT_EYE, (int)BodyParts.LEFT_EAR },
            new int[2]{ (int)BodyParts.RIGHT_EYE, (int)BodyParts.RIGHT_EAR },
            new int[2]{ (int)BodyParts.LEFT_SHOULDER, (int)BodyParts.RIGHT_SHOULDER },
            new int[2]{ (int)BodyParts.LEFT_SHOULDER, (int)BodyParts.LEFT_ELBOW },
            new int[2]{ (int)BodyParts.RIGHT_SHOULDER, (int)BodyParts.RIGHT_ELBOW },
            new int[2]{ (int)BodyParts.LEFT_ELBOW, (int)BodyParts.LEFT_WRIST },
            new int[2]{ (int)BodyParts.RIGHT_ELBOW, (int)BodyParts.RIGHT_WRIST },
            new int[2]{ (int)BodyParts.LEFT_SHOULDER, (int)BodyParts.LEFT_HIP },
            new int[2]{ (int)BodyParts.RIGHT_SHOULDER, (int)BodyParts.RIGHT_HIP },
            new int[2]{ (int)BodyParts.LEFT_HIP, (int)BodyParts.RIGHT_HIP }
        };

        /// <summary>
        /// Define a linear chain adjacency matrix between our keypoints <see cref="m_keypoints"/> using
        /// the indices from <see cref="BodyParts"/>,  allowing for a graph-like representation. 
        /// 
        /// It defines the parent->children relationship, which is a directed
        /// relationship as opposed to <see cref="m_keypointsJoints"/>. It can be used in reverse order
        /// to access the children->parent relationship.
        /// 
        /// It is useful in order to refine the estimated pose using the displacement vectors
        /// for multi-body pose estimation (also more robust than single body pose estimation 
        /// solely based on heatmap and offsets). We recall that the displacement vectors define 
        /// an approximate translation generated by the network between a parent and its children 
        /// (forward), and vice-versa (backward).
        /// As a result :
        /// Direct parent->children relationship could be used with the forward displacement
        /// vectors from <see cref="m_outputTensors[2]"/> to get human body pose starting at
        /// a root / parent node. 
        /// Reverse children->parent relationship could be used with the backward displacement
        /// vectors from <see cref="m_outputTensors[3]"/> to get human body pose starting at 
        /// a root / children node.
        /// </summary>
        public int[][] m_keypointsChain = new int[][]
        {
              new int[2]{ (int)BodyParts.NOSE, (int)BodyParts.LEFT_EYE },
              new int[2]{ (int)BodyParts.LEFT_EYE, (int)BodyParts.LEFT_EAR },
              new int[2]{ (int)BodyParts.NOSE, (int)BodyParts.RIGHT_EYE },
              new int[2]{ (int)BodyParts.RIGHT_EYE, (int)BodyParts.RIGHT_EAR },
              new int[2]{ (int)BodyParts.NOSE, (int)BodyParts.LEFT_SHOULDER },
              new int[2]{ (int)BodyParts.LEFT_SHOULDER, (int)BodyParts.LEFT_ELBOW },
              new int[2]{ (int)BodyParts.LEFT_ELBOW, (int)BodyParts.LEFT_WRIST },
              new int[2]{ (int)BodyParts.LEFT_SHOULDER, (int)BodyParts.LEFT_HIP },
              new int[2]{ (int)BodyParts.LEFT_HIP, (int)BodyParts.LEFT_KNEE },
              new int[2]{ (int)BodyParts.LEFT_KNEE, (int)BodyParts.LEFT_ANKLE },
              new int[2]{ (int)BodyParts.NOSE, (int)BodyParts.RIGHT_SHOULDER },
              new int[2]{ (int)BodyParts.RIGHT_SHOULDER, (int)BodyParts.RIGHT_ELBOW },
              new int[2]{ (int)BodyParts.RIGHT_ELBOW, (int)BodyParts.RIGHT_WRIST },
              new int[2]{ (int)BodyParts.RIGHT_SHOULDER, (int)BodyParts.RIGHT_HIP },
              new int[2]{ (int)BodyParts.RIGHT_HIP, (int)BodyParts.RIGHT_KNEE },
              new int[2]{ (int)BodyParts.RIGHT_KNEE, (int)BodyParts.RIGHT_ANKLE }
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

            // Populate our array of keypoints
            for (int i = 0; i < m_keypoints.Length; i++)
            {
                m_keypoints[i] = new Keypoint();
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
        /// Dispose our model <see cref="m_model"/> and interpreter <see cref="m_interpreter"/>.
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
        /// ToDo
        /// Perform a forward pass on the image using the current <see cref="m_interpreter"/>.
        /// We assume a PoseNetEstimator instance constructed with the constructor with arguments.
        /// Process:
        ///     -> convert inputImage to float32 precision (values 0...1) 
        ///     -> resize inputImage to input tensor dim and load it in input tensor
        ///     -> invoke interpreter to perform a forward pass, get heatmaps and offsets in output tensor dim
        ///     -> get keypoint from heatmaps in output tensor dim
        ///     -> translate keypoint to input tensor dim using offset 
        ///     -> rescale keypoints to inputImage dim
        ///     -> return keypoints
        /// </summary>
        /// <param name="inputImage">A RGB image. It will be resized during the inference to match the network's input size.</param>
        /// <returns>
        /// On error : An empty array of <see cref="Keypoint"/>.
        /// On success : Return an array of <see cref="m_numberOfKeypoints"/> <see cref="Keypoint"/> representing human body parts ordered as <see cref="BodyParts"/>.
        ///              If the probability <see cref="Keypoint.score"/> of a <see cref="Keypoint"/> is too low (hardcoded threshold for now, see below),
        ///              keypoint position is set to Point(-1,-1). This value can be used in conditional statements.
        ///              The keypoints are returned in the dimension of the inputImage. No need to further rescale the result for display. 
        ///          
        /// </returns>
        public Keypoint[] Inference(Emgu.CV.Mat inputImage)
        {
            // 0- Forward pass
            // Is the input empty ?
            if (inputImage.IsEmpty)
            {
                Console.WriteLine("ERROR:");
                Console.WriteLine("Empty image given to Inference PoseNetEstimarot. " +
                                  "Return new Keyoint[0] - empty array of Keypoints.");
                return new Keypoint[0];
            }

            int inputWidth  = inputImage.Cols;
            int inputHeigth = inputImage.Rows;
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
                    return new Keypoint[0];
                }
            }

            // 1- Converts 3D tensors to Emgu.CV.Mat
            Emgu.CV.Mat heatmaps_mat              = new Emgu.CV.Mat();
            Emgu.CV.Mat offsets_mat               = new Emgu.CV.Mat();
            Emgu.CV.Mat displacement_forward_mat  = new Emgu.CV.Mat();
            Emgu.CV.Mat displacement_backward_mat = new Emgu.CV.Mat();
            try
            {
                heatmaps_mat = new Mat(
                    m_outputTensors[0].Dims[1], m_outputTensors[0].Dims[2],
                    DepthType.Cv32F, m_outputTensors[0].Dims[3], m_outputTensors[0].DataPointer,
                    sizeof(float) * 3 * m_outputTensors[0].Dims[1]);
                offsets_mat = new Mat(
                    m_outputTensors[1].Dims[1], m_outputTensors[1].Dims[2],
                    DepthType.Cv32F, m_outputTensors[1].Dims[3], m_outputTensors[1].DataPointer,
                    sizeof(float) * 3 * m_outputTensors[1].Dims[1]);
                displacement_forward_mat = new Mat(
                    m_outputTensors[2].Dims[1], m_outputTensors[2].Dims[2],
                    DepthType.Cv32F, m_outputTensors[2].Dims[3], m_outputTensors[2].DataPointer,
                    sizeof(float) * 3 * m_outputTensors[2].Dims[1]);
                displacement_backward_mat = new Mat(
                    m_outputTensors[3].Dims[1], m_outputTensors[3].Dims[2],
                    DepthType.Cv32F, m_outputTensors[3].Dims[3], m_outputTensors[3].DataPointer,
                    sizeof(float) * 3 * m_outputTensors[3].Dims[1]);
            }
            catch
            {
                Console.WriteLine("Unable to read heatmaps or offsets in PoseNetEstimator. " +
                                  "Return new Keyoint[0] - empty array of Keypoints.");
                return new Keypoint[0];
            }

            // 2 - Split channels and store them in vector of mat
            if (!heatmaps_mat.IsEmpty & !offsets_mat.IsEmpty)
            {
                Emgu.CV.CvInvoke.Split(heatmaps_mat, m_heatmapsChannels);
                Emgu.CV.CvInvoke.Split(offsets_mat,m_offsetsChannels);
                Emgu.CV.CvInvoke.Split(displacement_forward_mat, m_forwardDisplacementChannels);
                Emgu.CV.CvInvoke.Split(displacement_backward_mat, m_backwardDisplacementChannels);
            }
            else
            {
                Console.WriteLine("Empty heatmaps_mat or offsets_mat in Inference from PoseNetEstimator. " +
                                  "Return new Keyoint[0] - empty array of Keypoints.");
                return new Keypoint[0];
            }

            // 3 -Estimate body pose
            //singleBodyPoseEstimation();
            improveSingleBodyPoseEstimation();
            rescaleKeypointsPosition(inputWidth, inputHeigth);

            return m_keypoints;
        }

        //ToDo
        void singleBodyPoseEstimation()
        {
            try
            {
                for (int keypointIndex = (int)BodyParts.NOSE; 
                         keypointIndex <= (int)BodyParts.RIGHT_WRIST; // only uper body parts 
                         keypointIndex++) 
                {
                    retrieveKeypointPositionFromHeatmap(keypointIndex);
                }
            }
            catch
            {
                Console.WriteLine("Error in PoseNetEstimator Inference : unable to decode heatmaps and offsets. " +
                                  "Return void from singleBodyPose in PoseNetEstimator.");

                return ;
            }
        }

        //ToDo
        void improveSingleBodyPoseEstimation(int root = (int)BodyParts.NOSE)
        {
            foreach(var kpt in m_keypoints)
            {
                kpt.reset();
            }

            if (root >= (int)BodyParts.NOSE & root < (int)BodyParts.NUMBER_MAX)
            {
                retrieveKeypointPositionFromHeatmap(keypointIndex: root, 
                                                    inInputTensorDim: true,
                                                    withOffset: true);


                // Iterate over the linear chain of parent->children
                // Decode the part positions upwards in the tree, following the backward
                // displacements.
                for (var edge = 0; edge < m_keypointsChain.Length - 1; edge++)
                {
                    var sourceKeypointId = m_keypointsChain[edge][0];
                    var targetKeypointId = m_keypointsChain[edge][1];
                    if (m_keypoints[targetKeypointId].position == new Point(-1, -1))
                    {
                        // With position_raw
                        int[] displacement = getForwardDisplacement(edge, m_keypoints[sourceKeypointId].position_raw);
                        int[] source_offset = getOffset(sourceKeypointId, m_keypoints[sourceKeypointId].position_raw);
                        int output_stride = (m_inputTensor.Dims[1] - 1) / (m_outputTensors[0].Dims[1] - 1);

                        // Position is based on source position, offset and displacement all in inputscale.
                        // Raw position rescale the target position.
                        m_keypoints[targetKeypointId].position =
                            new Point(m_keypoints[sourceKeypointId].position.X - source_offset[0] + displacement[0],
                                      m_keypoints[sourceKeypointId].position.Y - source_offset[1] + displacement[1]);
                        m_keypoints[targetKeypointId].position_raw =
                            new Point((m_keypoints[targetKeypointId].position.X) / output_stride,
                                      (m_keypoints[targetKeypointId].position.Y) / output_stride);

                        LocalMax(targetKeypointId, true, true);
                    }
                }
                for (var edge = m_keypointsChain.Length - 1; edge >= 0; edge--)
                {
                    var sourceKeypointId = m_keypointsChain[edge][1];
                    var targetKeypointId = m_keypointsChain[edge][0];
                    if (m_keypoints[targetKeypointId].position == new Point(-1, -1))
                    {
                        // With position_raw
                        int[] displacement = getBackwardDisplacement(edge, m_keypoints[sourceKeypointId].position_raw);
                        int[] source_offset = getOffset(sourceKeypointId,
                                                        m_keypoints[sourceKeypointId].position_raw);
                        int output_stride = (m_inputTensor.Dims[1] - 1) / (m_outputTensors[0].Dims[1] - 1);

                        // Position is based on source position, offset and displacement all in inputscale.
                        // Raw position rescale the target position.
                        m_keypoints[targetKeypointId].position =
                            new Point(m_keypoints[sourceKeypointId].position.X - source_offset[0] + displacement[0],
                                      m_keypoints[sourceKeypointId].position.Y - source_offset[1] + displacement[1]);
                        m_keypoints[targetKeypointId].position_raw =
                            new Point((m_keypoints[targetKeypointId].position.X) / output_stride,
                                      (m_keypoints[targetKeypointId].position.Y) / output_stride);

                        LocalMax(targetKeypointId, true, true);
                    }
                }


            }
        }

        //ToDo
        void retrieveKeypointPositionFromHeatmap(int keypointIndex, bool inInputTensorDim = true, bool withOffset = true)
        {
            // Casual check to warn the user about misuse
            if (withOffset & !inInputTensorDim)
            {
                Console.WriteLine("/!\\ Warning in retrievekeypointPosition from PoseNet Estimator." +
                                  " The withOffset flag set to true while inInputTensorDim flag is false. " +
                                  " Offset will not be applied (offset values are in input tensor dimensions). ");
            }

            // Do not consider output of body parts indexes to avoid out of memory access
            if (keypointIndex >= (int)BodyParts.NOSE & keypointIndex < (int)BodyParts.NUMBER_MAX) //valid
            {
                // Find point with highest probability to be a body part in the corresponding heatmap channel
                var maxLoc = new Point();
                var minLoc = new Point();
                double min = 0;
                double max = 0;
                Emgu.CV.CvInvoke.MinMaxLoc(m_heatmapsChannels[keypointIndex], ref min, ref max, ref minLoc, ref maxLoc);

                // Update score, body part and location. 
                m_keypoints[keypointIndex].score = (float)sigmoid(max); // We apply sigmoid on the max value to get [0...1] probability score.
                m_keypoints[keypointIndex].bodyPart = keypointIndex;
                if (m_keypoints[keypointIndex].score > 0.05) // 0.05 is a fixed probability threshold between 0 and 1 - todo: remove hardcoded
                {
                    // Retrieve keypoint position and offset.
                    m_keypoints[keypointIndex].position = maxLoc;
                    m_keypoints[keypointIndex].position_raw = maxLoc;

                    // Scale to input dim using output_stride, then offset using offset values
                    if (inInputTensorDim)
                    {
                        int output_stride = (m_inputTensor.Dims[1] - 1) / (m_outputTensors[0].Dims[1] - 1);
                        m_keypoints[keypointIndex].position.X *= output_stride;
                        m_keypoints[keypointIndex].position.Y *= output_stride;

                        if (withOffset)
                        {
                            int[] offset = getOffset(keypointIndex, m_keypoints[keypointIndex].position_raw);
                            m_keypoints[keypointIndex].position.X += offset[0];
                            m_keypoints[keypointIndex].position.Y += offset[1];
                        }
                    }

                }
                else
                {
                    m_keypoints[keypointIndex].position = new Point(-1, -1);
                }
            }
        }

        // ToDo
        private void LocalMax(int keypointIndex, bool inInputTensorDim = true, bool withOffset = true)
        {
            // Casual check to warn the user about misuse
            if (withOffset & !inInputTensorDim)
            {
                Console.WriteLine("/!\\ Warning in retrievekeypointPosition from PoseNet Estimator." +
                                  " The withOffset flag set to true while inInputTensorDim flag is false. " +
                                  " Offset will not be applied (offset values are in input tensor dimensions). ");
            }

            // Do not consider output of body parts indexes to avoid out of memory access
            if (keypointIndex >= (int)BodyParts.NOSE & keypointIndex < (int)BodyParts.NUMBER_MAX) //valid
            {
                var maxLoc = new Point();
                float max = -1;

                // Find point with highest probability to be a body part in the corresponding heatmap channel
                for (int i = -2; i <= 2; i++)
                {
                    for (int j = -2; j <= 2; j++)
                    {
                        if (m_keypoints[keypointIndex].position_raw.Y + i > 0 & m_keypoints[keypointIndex].position_raw.Y + i < m_heatmapsChannels[keypointIndex].Rows &
                           m_keypoints[keypointIndex].position_raw.X + j > 0 & m_keypoints[keypointIndex].position_raw.X + j < m_heatmapsChannels[keypointIndex].Cols)
                        {
                            var val = m_heatmapsChannels[keypointIndex].GetValue(m_keypoints[keypointIndex].position_raw.Y + i,
                                                                                 m_keypoints[keypointIndex].position_raw.X + j);
                            if (val > max)
                            {
                                max = val;
                                maxLoc.X = m_keypoints[keypointIndex].position_raw.X + j;
                                maxLoc.Y = m_keypoints[keypointIndex].position_raw.Y + i;
                            }
                        }
                    }
                }
                // Update score, body part and location. 
                m_keypoints[keypointIndex].score = (float)sigmoid(max); // We apply sigmoid on the max value to get [0...1] probability score.
                m_keypoints[keypointIndex].bodyPart = keypointIndex;
                if (m_keypoints[keypointIndex].score > 0)
                {
                    m_keypoints[keypointIndex].position = maxLoc;
                    m_keypoints[keypointIndex].position_raw = maxLoc;
                }
                else
                {
                    m_keypoints[keypointIndex].reset();
                }

                // Scale to input dim using output_stride, then offset using offset values
                if (inInputTensorDim & maxLoc.X != -1 & maxLoc.Y != -1)
                {
                    int output_stride = (m_inputTensor.Dims[1] - 1) / (m_outputTensors[0].Dims[1] - 1);
                    m_keypoints[keypointIndex].position.X *= output_stride;
                    m_keypoints[keypointIndex].position.Y *= output_stride;

                    if (withOffset)
                    {
                        int[] offset = getOffset(keypointIndex, m_keypoints[keypointIndex].position_raw);
                        m_keypoints[keypointIndex].position.X += offset[0];
                        m_keypoints[keypointIndex].position.Y += offset[1];
                    }
                }
            }
        }

        //ToDo
        private int[] getOffset(int keypointIndex, Point positionInOutputDim)
        {
            if (positionInOutputDim.Y < m_offsetsChannels[keypointIndex].Cols & positionInOutputDim.Y > 0 &
                positionInOutputDim.X < m_offsetsChannels[keypointIndex].Rows & positionInOutputDim.X > 0)
            {
                Image<Gray, Single> offset_y = m_offsetsChannels[keypointIndex].ToImage<Gray, Single>();
                Image<Gray, Single> offset_x = m_offsetsChannels[keypointIndex + m_numberOfKeypoints].ToImage<Gray, Single>();
                var y = offset_y[positionInOutputDim.Y, positionInOutputDim.X];
                var x = offset_x[positionInOutputDim.Y, positionInOutputDim.X];
                return new int[2] { (int)x.Intensity, (int)y.Intensity };
            }
            return new int[2] { 0, 0 };
        }

        /// <summary>
        /// Return forward displacement vector in <see cref="m_outputTensors"/> dimension.
        /// </summary>
        /// <param name="keypointIndex">ToDo</param>
        /// <param name="positionInOutputDim">ToDo</param>
        /// <returns></returns>
        private int[] getForwardDisplacement(int edgeIndex, Point positionInOutputDim)
        {
            if (positionInOutputDim.Y < m_offsetsChannels[edgeIndex].Cols & positionInOutputDim.Y > 0 &
                positionInOutputDim.X < m_offsetsChannels[edgeIndex].Rows & positionInOutputDim.X > 0)
            {
                Image<Gray, Single> displacement_y = m_forwardDisplacementChannels[edgeIndex].ToImage<Gray, Single>();
                Image<Gray, Single> displacement_x = m_forwardDisplacementChannels[edgeIndex + (m_numberOfKeypoints - 1)].ToImage<Gray, Single>();
                var y = displacement_y[positionInOutputDim.Y, positionInOutputDim.X];
                var x = displacement_x[positionInOutputDim.Y, positionInOutputDim.X];
                return new int[2] { (int)x.Intensity, (int)y.Intensity };
            }
            return new int[2] { 0, 0 };
        }

        /// <summary>
        /// Return backward displacement vector in <see cref="m_outputTensors"/> dimension.
        /// </summary>
        /// <param name="keypointIndex">ToDo</param>
        /// <param name="positionInOutputDim">ToDo</param>
        /// <returns></returns>
        private int[] getBackwardDisplacement(int edgeIndex, Point positionInOutputDim)
        {
            if (positionInOutputDim.Y < m_offsetsChannels[edgeIndex].Cols & positionInOutputDim.Y > 0 &
                positionInOutputDim.X < m_offsetsChannels[edgeIndex].Rows & positionInOutputDim.X > 0)
            {
                Image<Gray, Single> displacement_y = m_backwardDisplacementChannels[edgeIndex].ToImage<Gray, Single>();
                Image<Gray, Single> displacement_x = m_backwardDisplacementChannels[edgeIndex + (m_numberOfKeypoints - 1)].ToImage<Gray, Single>();
                var y = displacement_y[positionInOutputDim.Y, positionInOutputDim.X];
                var x = displacement_x[positionInOutputDim.Y, positionInOutputDim.X];
                return new int[2] { (int)x.Intensity, (int)y.Intensity };
            }
            return new int[2] { 0, 0 };
        }

        /// <summary>
        /// Rescale keypoints position from <see cref="m_inputTensor"/> dimensions to
        /// another spatial dimension (e.g., input image dimension before it being resize
        /// to match inputTensor dimension). Following formulas are used:
        /// newX = (currentX / currentWidth) * newWidth
        /// newY = (currentY / currentHeight) * newHeight
        /// HINT: Very useful for display purpose.
        /// </summary>
        /// <param name="newWidth">Output dimension width</param>
        /// <param name="newHeigth">Output dimension heigth</param>
        void rescaleKeypointsPosition(int newWidth, int newHeigth)
        {
            foreach(var kpt in m_keypoints)
            {
                if (kpt.position.X != -1 & kpt.position.Y != -1 & newWidth > 0 & newHeigth > 0) //valid
                {
                    kpt.position.X = kpt.position.X * newWidth / m_inputTensor.Dims[2];
                    kpt.position.Y = kpt.position.Y * newHeigth / m_inputTensor.Dims[1];
                }
            }
        }

        /// <summary>
        /// Rescale a keypoint position from <see cref="m_inputTensor"/> dimensions to
        /// another spatial dimension (e.g., input image dimension before it being resize
        /// to match inputTensor dimension). Following formulas are used:
        /// newX = (currentX / currentWidth) * newWidth
        /// newY = (currentY / currentHeight) * newHeight
        /// HINT: Very useful for display purpose.
        /// </summary>
        /// <param name="keypointIndex">Index of the keypoint in <see cref="m_keypoints"/>. This value should
        /// be higher or equal to <see cref="BodyParts.NOSE"/> and strickly lower than <see cref="BodyParts.NUMBER_MAX"/></param>
        /// <param name="newWidth">Output dimension width</param>
        /// <param name="newHeigth">Output dimension heigth</param>
        void rescaleKeypointsPosition(int keypointIndex, int newWidth, int newHeigth)
        {
            if (keypointIndex >= (int)BodyParts.NOSE &
               keypointIndex < (int)BodyParts.NUMBER_MAX) //valid
            {
                if (m_keypoints[keypointIndex].position.X != -1 &
                    m_keypoints[keypointIndex].position.Y != -1 &
                    newWidth > 0 & newHeigth > 0) //valid
                {
                    m_keypoints[keypointIndex].position.X =
                        m_keypoints[keypointIndex].position.X * newWidth / m_inputTensor.Dims[2];
                    m_keypoints[keypointIndex].position.Y =
                        m_keypoints[keypointIndex].position.Y * newHeigth / m_inputTensor.Dims[1];
                }
            }
        }
    }
}
