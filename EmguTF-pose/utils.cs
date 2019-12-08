using System;
using System.Drawing;
using Emgu.TF.Lite;
using Emgu.CV;

namespace EmguTF_pose
{
    /// <summary>
    /// A static class that contains very useful utilities to manipulate images and tensors.
    /// </summary>
    static class Utils
    {
        /// <summary>
        /// Public interface to convert a 3 channel BGR Mat to a Tensor.
        /// BGR Mat are Emgu.CV.Mat objects. Tensor are Emgu.TF.Lite.Tensor objects.
        /// NB: This function will resize the BGR mat to match the Tensor spatial dimensions if necessary.
        /// 
        /// Stolen from https://github.com/emgucv/emgutf/blob/master/Emgu.TF.Example/CVInterop/TensorConvert.cs
        /// </summary>
        /// <param name="image">The input Emgu CV Mat</param>
        /// <param name="tensor">The pre-allocated output tensor. Dimension must match (1, inputHeight, inputWidth, 3)</param>
        /// <param name="inputHeight">The height of the image in the output tensor, if it is -1, the height will not be changed.</param>
        /// <param name="inputWidth">The width of the image in the output tensor, if it is -1, the width will not be changed.</param>
        /// <param name="inputMean">The mean, if it is not 0, the value will be substracted from the pixel values</param>
        /// <param name="scale">The optional scale</param>
        /// <returns>The tensorflow tensor</returns>
        public static void ReadTensorFromMatBgr(Mat image, Tensor tensor, int inputHeight = -1, int inputWidth = -1, float inputMean = 0.0f, float scale = 1.0f)
        {
            if (image.NumberOfChannels != 3)
            {
                throw new ArgumentException("Input must be 3 channel BGR image");
            }

            Emgu.CV.CvEnum.DepthType depth = image.Depth;
            if (!(depth == Emgu.CV.CvEnum.DepthType.Cv8U || depth == Emgu.CV.CvEnum.DepthType.Cv32F))
            {
                throw new ArgumentException("Input image must be 8U or 32F");
            }

            //resize
            int finalHeight = inputHeight == -1 ? image.Height : inputHeight;
            int finalWidth = inputWidth == -1 ? image.Width : inputWidth;
            Size finalSize = new Size(finalWidth, finalHeight);

            int[] dim = tensor.Dims;
            if (dim[0] != 1)
                throw new ArgumentException("First dimension of the tensor must be 1 (batch size)");

            if (dim[1] != finalHeight)
                throw new ArgumentException("Second dimension of the tensor must match the input height");

            if (dim[2] != finalWidth)
                throw new ArgumentException("Third dimension of the tensor must match the input width");

            if (dim[3] != 3)
                throw new ArgumentException("Fourth dimension of the tensor must be 3 (BGR has 3 channels)");

            if (image.Size != finalSize)
            {
                using (Mat tmp = new Mat())
                {
                    CvInvoke.Resize(image, tmp, finalSize);
                    ReadTensorFromMatBgr(tmp, inputMean, scale, tensor);
                    tmp.Dispose();
                }
            }
            else
            {
                ReadTensorFromMatBgr(image, inputMean, scale, tensor);
            }
        }

        /// <summary>
        /// Private interface to convert a BGR Mat image to a Tensor.
        /// Actually perfom Emgu.CV.Mat to Emgu.TF.Lite.Tensor conversion.
        /// Stolen from https://github.com/emgucv/emgutf/blob/master/Emgu.TF.Example/CVInterop/TensorConvert.cs
        /// </summary>
        /// <param name="image">The input Emgu CV Mat</param>
        /// <param name="inputMean">The mean, if it is not 0, the value will be substracted from the pixel values</param>
        /// <param name="scale">The optional scale</param>
        /// <param name="t">The pre-allocated output tensor. Dimension must match (1, inputHeight, inputWidth, 3)</param>
        /// <returns>The tensorflow tensor</returns>
        private static void ReadTensorFromMatBgr(Mat image, float inputMean, float scale, Tensor t)
        {
            DataType type = t.Type;
            if (type == DataType.Float32)
            {
                using (Mat matF = new Mat(image.Size, Emgu.CV.CvEnum.DepthType.Cv32F, 3, t.DataPointer, sizeof(float) * 3 * image.Width))
                {
                    image.ConvertTo(matF, Emgu.CV.CvEnum.DepthType.Cv32F);
                }
            }
            else if (type == DataType.UInt8)
            {
                using (Mat matB = new Mat(image.Size, Emgu.CV.CvEnum.DepthType.Cv8U, 3, t.DataPointer, sizeof(byte) * 3 * image.Width))
                {
                    if (scale == 1.0f && inputMean == 0)
                    {
                        image.CopyTo(matB);
                    }
                    else
                        CvInvoke.ConvertScaleAbs(image, matB, scale, -inputMean * scale);
                }
            }
            else
            {
                throw new Exception(String.Format("Data Type of {0} is not supported.", type));
            }
        }

    }
}
