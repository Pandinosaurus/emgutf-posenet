using System;
using System.IO;
using System.Runtime.ExceptionServices;
using System.Security;
using Emgu.CV;

namespace EmguTF_pose
{
    /// <summary>
    /// This class represents a deep neural network loaded with Emgu.TF.Lite
    /// </summary>
    class DeepNetworkLite
    {
        /// <summary>
        /// Interpreter allowing to load and use tensorflow lite frozen model.
        /// It is a network abstraction in Emgu.TF.
        /// </summary>
        protected Emgu.TF.Lite.Interpreter m_interpreter;

        /// <summary>
        /// A path to a frozen deep neural network saved with tensorflow lite.
        /// </summary>
        protected String m_frozenModelPath;

        /// <summary>
        /// Expected extension for a frozen model (.tflite)
        /// </summary>
        protected String m_expectedModelExtension = ".tflite";

        /// <summary>
        /// Number of CPU threads the network / interpeter will use.
        /// </summary>
        protected int m_numberOfThreads { get; set; }

        /// <summary>
        /// Default constructor. It does nothing but allocating memory. 
        /// You need to specify a frozen model path to make it works.
        /// TIP : use the constructor with arguments, this one is useless for now.
        /// </summary>
        public DeepNetworkLite() { }

        /// <summary>
        /// Constructor with arguments. You should use this one.
        /// </summary>
        /// <param name="frozenModelPath">Path to a deep neural network frozen and saved with tensorflow lite.</param>
        /// <param name="numberOfThreads">Number of threads the neural network will be able to use (default: 2)</param>
        public DeepNetworkLite(String frozenModelPath, int numberOfThreads = 2) {
            // Check file
            if (!File.Exists(frozenModelPath))
            {
                Console.WriteLine("ERROR:");
                Console.WriteLine("FrozenModelPath specified in DeepNetworkLite " +
                                  "construtor with argument does not exist.");
                Console.WriteLine("Network not loaded.");
                return;
            }
            if(Path.GetExtension(frozenModelPath) != m_expectedModelExtension)
            {
                Console.WriteLine("ERROR:");
                Console.WriteLine("Extension of specified frozen model path in DeepNetworkLite " +
                                  "constructor with argument does not" +
                                  "match " + m_expectedModelExtension);
                Console.WriteLine("Network not loaded.");
                return;
            }
            m_frozenModelPath = frozenModelPath;

            try
            {
                Emgu.TF.Lite.FlatBufferModel flatbuffer;
                flatbuffer = new Emgu.TF.Lite.FlatBufferModel(filename: m_frozenModelPath);
                m_interpreter = new Emgu.TF.Lite.Interpreter(flatBufferModel: flatbuffer);
                m_interpreter.AllocateTensors();
                m_interpreter.SetNumThreads(numThreads: numberOfThreads);
                flatbuffer.Dispose();
            }
            catch
            {
                m_frozenModelPath = ""; // reset
                Console.WriteLine("ERROR:");
                Console.WriteLine("Unable to load frozen model in DeepNetworkLite constructor with arguments " +
                                  "despite files was found with correct extension. " +
                                  "Please, make sure you saved your model using tensorflow lite pipelines.");
                return;
            }
            return;
        }

        ~DeepNetworkLite()
        {
            if (m_interpreter.Ptr != IntPtr.Zero) //in case garbage collector messed up
            {
                m_interpreter.Dispose();
            }
        }

        /// <summary>
        /// Perform a forward pass on an image using the interpreter / deep neural network.
        /// </summary>
        /// <param name="image">An Emgu.CV.Mat image in BGR format.</param>
        /// <returns>An array of Emgu.TF.Lite.Tensor containing the outputs of the network.</returns>
        [HandleProcessCorruptedStateExceptions]
        [SecurityCritical]
        protected Emgu.TF.Lite.Tensor[] InferenceOnImage(Emgu.CV.Mat image)
        {

                // Is the input empty ?
                if (image.IsEmpty)
                {
                    Console.WriteLine("ERROR:");
                    Console.WriteLine("Empty image given to InferenceOnImage in DeepNetworkLite classe. " +
                                        "Return Emgu.TF.Lite.Tensor[0].");
                    return new Emgu.TF.Lite.Tensor[0];
                }
                // Is the input continuous in memory ?
                image = image.Clone();

                // Is the input encoded with 32 bit floating point precision ?
                if (image.Depth != Emgu.CV.CvEnum.DepthType.Cv32F)
                {
                    image.ConvertTo(image, Emgu.CV.CvEnum.DepthType.Cv32F);
                    image /= 255;
                }
                if (image.DataPointer != null)
                {
                    try
                    {
                        // Load image in interpreter using ReadTensorFromMatBgr function from the utils.
                        Utils.ReadTensorFromMatBgr(
                             image: image,
                             tensor: m_interpreter.Inputs[0],
                             inputHeight: m_interpreter.Inputs[0].Dims[1],
                             inputWidth: m_interpreter.Inputs[0].Dims[2]
                         );

                        // Actually perfom the inference
                        m_interpreter.Invoke();
                        GC.KeepAlive(m_interpreter); 
                    }
                    catch // The garbage collector messed up - probably due to an error in EmguCV wrapper to tensorflow C++ code
                    {
                        Console.WriteLine("Error in DeepNetworkLite Inference");
                        Console.WriteLine("Disposing...");
                        try
                        {
                            Console.WriteLine("Disposing interpreter...");
                            if (m_interpreter.Ptr != IntPtr.Zero) //in case garbage collector messed up
                            {
                                m_interpreter.Dispose();
                                GC.KeepAlive(m_interpreter);
                             }
                        }
                        catch
                        {
                            Console.WriteLine("Unable to dispose the interpreter.");
                        }
                        try
                        {
                            Console.WriteLine("Disposing image...");
                            image.Dispose();
                        }
                        catch
                        {
                            Console.WriteLine("Unable to dispose the image.");
                        }

                        Console.WriteLine("Recreate interpretor...");
                        Emgu.TF.Lite.FlatBufferModel flatbuffer;
                        flatbuffer = new Emgu.TF.Lite.FlatBufferModel(filename: m_frozenModelPath);
                        m_interpreter = new Emgu.TF.Lite.Interpreter(flatBufferModel: flatbuffer);
                        m_interpreter.AllocateTensors();
                        m_interpreter.SetNumThreads(numThreads: 4);

                        Console.WriteLine("Return empty outputs...");
                        GC.KeepAlive(m_interpreter);
                        return m_interpreter.Outputs;
                    }
                }

                // Return the output after inference
                image.Dispose();
                GC.KeepAlive(m_interpreter);
                return m_interpreter.Outputs;
        }

        /// <summary>
        /// Apply sigmoid on a value
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }

    }
}
