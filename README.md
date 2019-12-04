# EmguTF-PoseNet
Deep pose estimation in C# using tensorflow lite through Emgu.TF on CPU. <br>
We achieve ~8-9 FPS on an intel i7-7600U @ 2.80Ghz (4 CPUs).
<br> <br>
![alt text](/docs/demo.gif)
<br>

## Why
Emgu.TF is rather new, and it is one of the rare libraries providing a C# wrapper for tensorflow lite. Official exemples in Emgu.TF demonstrate how to classify and detect objects on static images. In this work, the focus is put on human pose estimation from a webcam flow, which comes with its own difficulties. 

## Network
We will reuse the PoseNet weights provided by the tensorflow team and optimized for fast inference with tensorflow lite. 

-- Section ToDo ---

## Code use steps
Keep in mind this work is a working prototype, that may contain bugs.

1. Make sure you are working with Visual Studio 2017 (my version is Community) on Windows 10.
2. Clone this github repository: ``` git clone https://github.com/Pandinosaurus/EmguTF-PoseNet/ ```. All dll should be already included. We will assume you have clone the repository in ```<pathto>/EmguTF-PoseNet```, where ```<pathto>``` is your current location and ```EmguTF-PoseNet``` is the downloaded repository. 
3. Go into the EmguTF-PoseNet directory, and double click on the .sln file (```EmguTF-pose.sln```). Visual studio should open the solution.
4. Select x64 as the target plateform in the top panel. It is located near the "Debug"/"Release" combobox.
5. Open the ```Form1.cs``` file, and modify the line 103 to make the ```frozenModelPath``` to match yours. It should be ```<pathto>/EmguTF-PoseNet/models/```.
6. Clean the project: right click on EmguTF-pose in the solution explorer -> clean
7. Regenerate the project: right click on EmguTF-pose in the solution explorer -> regenerate 
8. You may now be able to launch the application clicking on the green arrow in the top panel. If you encounter any bug and find solutions to them, please, feel free to share them. 

NB: We recommend doing step 6 and 7 everytime you would like to launch the application from Visual Studio (not using the .exe).
