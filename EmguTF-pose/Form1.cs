using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV; //VideoCapture, Mat 
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;


namespace EmguTF_pose
{ 

    public partial class EmguTF_posewindow : Form
    {
        /// <summary>
        /// Our video m_Webcam object. Abstraction for our m_Webcam.
        /// </summary>
        private VideoCapture m_Webcam;

        /// <summary>
        /// Our current m_frame to process.
        /// </summary>
        private Mat m_frame;

        /// <summary>
        /// Our pose estimator
        /// </summary>
        private PoseNetEstimator m_posenet;

        /// <summary>
        /// A basic flag checking if process frame is ongoing or not.
        /// </summary>
        static bool inprocessframe = false;

        /// <summary>
        /// A basic flag checking if process is ongoing or not
        /// </summary>
        static bool inprocess = false;

        /// <summary>
        /// __init__
        /// </summary>
        public EmguTF_posewindow()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Everything start and stop on a button click.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button_start_Click(object sender, EventArgs e)
        {
            if (button_start.Text != "Stop")
            {
                button_start.Text = "Stop";
                m_Webcam.Start();
            }
            else
            {
                m_Webcam.Stop();
                button_start.Text = "Start";
            }
        }

        /// <summary>
        /// Called on load event. It will instantiate this class variables, including
        /// the callbacks functions.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void EmguTF_posewindow_Load(object sender, EventArgs e)
        {
            // Main Emgu CV elements to work with m_Webcam
            // 1- An matrix / image to store the last grabbed frame
            m_frame = new Mat();

            // 2- Our webcam will represent the first camera found on our device
            m_Webcam = new VideoCapture(0);

            // 3- When the webcam capture (grab) an image, callback on ProcessFrame method
            m_Webcam.ImageGrabbed += new EventHandler(Process); // event based

            // Pose estimator - to do remove hardcoded things !
            m_posenet = new PoseNetEstimator(frozenModelPath: "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite",
                                             numberOfThreads: 4);

        }

        /// <summary>
        /// Handle the image grabbed event by retriving the frame, processing it and showing it.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Process(object sender, EventArgs e)
        {
            if (!inprocess)
            {
                // Say we start to process
                inprocess = true;

                // Reinit m_frame
                m_frame = new Mat();

                // Retrieve
                m_Webcam.Retrieve(m_frame);

                // If frame is not empty, try to process it
                if (!m_frame.IsEmpty)
                {
                    //if not already processing previous frame, process it
                    if (!inprocessframe)
                    {
                        ProcessFrame(m_frame.Clone());
                    }

                    // Display keypoints and frame in imageview
                    ShowKeypoints();
                    ShowFrame();

                }

                // Say we finished to process 
                m_frame.Dispose();
                inprocess = false;
            }
        }

        /// <summary>
        /// A method to get keypoints from a frame.
        /// </summary>
        /// <param name="frame">A copy of <see cref="m_frame"/>. It could be resized beforehand.</param>
        public void ProcessFrame(Emgu.CV.Mat frame)
        {
            if (!inprocessframe)
            {
                inprocessframe = true;
                DateTime start = DateTime.Now;

                m_posenet.Inference(frame);

                DateTime stop = DateTime.Now;
                long elapsedTicks = stop.Ticks - start.Ticks;
                TimeSpan elapsedSpan = new TimeSpan(elapsedTicks);
                Console.WriteLine(1000/(double)elapsedSpan.Milliseconds);

                inprocessframe = false;
            }
        }

        /// <summary>
        /// Show<see cref="m_keypoints"/> on <see cref="m_frame"/> if keypoint is not Point(-1,-1).
        /// </summary>
        private void ShowKeypoints()
        {
            if (!m_frame.IsEmpty)
            {
                float count = 1;
                foreach (Point pt in m_posenet.m_keypoints) // if not empty array of points
                {
                    if ((pt.X != -1) & (pt.Y != -1)) // if points are valids
                    {
                        //newX = (currentX / currentWidth) * newWidth
                        //newY = (currentY / currentHeight) * newHeight
                        Emgu.CV.CvInvoke.Circle(m_frame,
                                                new Point((int)((float)pt.X * m_frame.Width / 257),
                                                            (int)((float)pt.Y * m_frame.Height / 257)),
                                                3, new MCvScalar(200, 255, (int)((float)255 / count), 255), 2);
                    }
                    count++;
                }
            }
        }

        /// <summary>
        /// Show <see cref="m_frame"/> on an ImageBox.
        /// </summary>
        private void ShowFrame()
        {
            if (!m_frame.IsEmpty)
            {
                CvInvoke.Resize(m_frame, m_frame, new Size(imageBox.Size.Width, imageBox.Size.Height));
                Emgu.CV.CvInvoke.Flip(m_frame, m_frame, FlipType.Horizontal);
                imageBox.Image = m_frame;
                Emgu.CV.CvInvoke.WaitKey(10); //wait a few clock cycles
            }
        }
    }
}
