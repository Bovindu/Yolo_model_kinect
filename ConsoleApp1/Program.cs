using Microsoft.Kinect;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.Collections.Generic;

class Program
{
    static KinectSensor _sensor;
    static RequestSocket _zmqSocket;
    static short[] _depthData;
    static byte[] _colorData;
    static int _depthWidth;
    static int _depthHeight;

    static void Main()
    {
        _zmqSocket = new RequestSocket();
        _zmqSocket.Connect("tcp://localhost:5555");

        _sensor = KinectSensor.KinectSensors.FirstOrDefault();
        if (_sensor != null)
        {
            // Enable streams
            _sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
            _sensor.DepthStream.Enable(DepthImageFormat.Resolution640x480Fps30);

            // Allocate space for frames
            _depthData = new short[_sensor.DepthStream.FramePixelDataLength];
            _colorData = new byte[_sensor.ColorStream.FramePixelDataLength];

            _sensor.AllFramesReady += AllFramesReady;
            _sensor.Start();

            Console.WriteLine("Kinect streaming started. Press Enter to exit.");
            Console.ReadLine();

            _sensor.Stop();
            _zmqSocket.Dispose();
            Cv2.DestroyAllWindows();
        }
    }

    static void AllFramesReady(object sender, AllFramesReadyEventArgs e)
    {
        using (ColorImageFrame colorFrame = e.OpenColorImageFrame())
        using (DepthImageFrame depthFrame = e.OpenDepthImageFrame())
        {
            if (colorFrame == null || depthFrame == null) return;

            // Get color data
            colorFrame.CopyPixelDataTo(_colorData);

            // Get depth data
            depthFrame.CopyPixelDataTo(_depthData);
            _depthWidth = depthFrame.Width;            
            _depthHeight = depthFrame.Height;            

            // Convert color data to bitmap and send to YOLO (existing code)
            Bitmap bmp = new Bitmap(colorFrame.Width, colorFrame.Height, PixelFormat.Format32bppRgb);
            BitmapData bmpData = bmp.LockBits(
                new Rectangle(0, 0, bmp.Width, bmp.Height),
                ImageLockMode.WriteOnly,
                bmp.PixelFormat
            );
            Marshal.Copy(_colorData, 0, bmpData.Scan0, _colorData.Length);
            bmp.UnlockBits(bmpData);

            Mat cvMat = BitmapConverter.ToMat(bmp);

            // Send to YOLO server (existing code)
            byte[] jpegBytes;
            using (MemoryStream ms = new MemoryStream())
            {
                bmp.Save(ms, ImageFormat.Jpeg);
                jpegBytes = ms.ToArray();
            }
            _zmqSocket.SendFrame(jpegBytes);

            string jsonResponse = _zmqSocket.ReceiveFrameString();
            var detections = JsonConvert.DeserializeObject<List<Detection>>(jsonResponse);

            // Create depth visualization
            Mat depthColorMat = RenderDepthFrame(_depthData, _depthWidth, _depthHeight);

            // Add these camera intrinsic parameters at the class level
            const double FX = 594.21;  // Focal length X (pixels)
            const double FY = 591.04;  // Focal length Y (pixels)
            const double CX = 339.5;   // Principal point X
            const double CY = 242.7;   // Principal point Y

            foreach (var d in detections)
            {
                var centerX = (d.BBox[0] + d.BBox[2]) / 2;                
                var centerY = (d.BBox[1] + d.BBox[3]) / 2;                

                // Map color coordinates to depth
                int depthX = (int)(centerX * _depthWidth / colorFrame.Width) - 13;                
                int depthY = (int)(centerY * _depthHeight / colorFrame.Height) - 10;
                int depthIndex = depthY * _depthWidth + depthX;

                if (depthIndex >= 0 && depthIndex < _depthData.Length)
                {
                    short depthValue = _depthData[depthIndex];
                    int depthInMm = depthValue >> DepthImageFrame.PlayerIndexBitmaskWidth;

                    // Draw on depth frame
                    Cv2.Circle(
                        depthColorMat, 
                        new OpenCvSharp.Point(depthX , depthY), 
                        5, Scalar.Red, 
                        -1
                    );

                    if (depthInMm > 0)  // Valid depth
                    {
                        // Convert to meters for calculations
                        double z = depthInMm / 1000.0;

                        // Calculate real-world coordinates
                        double x = (centerX - CX) * z / FX;
                        double y = (centerY - CY) * z / FY;

                        // Draw coordinate text
                        string coordText = $"X: {x:F2}m  Y: {y:F2}m  Z: {z:F2}m";
                        Cv2.PutText(
                            cvMat,
                            coordText,
                            new OpenCvSharp.Point(centerX, centerY + 20),
                            HersheyFonts.HersheySimplex,
                            0.6,
                            Scalar.White,
                            1
                        );
                    }

                    Cv2.PutText(
                        depthColorMat,
                        $"{depthInMm}mm",
                        new OpenCvSharp.Point(depthX, depthY - 30),
                        HersheyFonts.HersheySimplex,
                        0.6,
                        Scalar.White,
                        1
                    );
                }
            }

            // Show frames
            Cv2.ImShow("RGB Frame", cvMat);
            Cv2.ImShow("Depth Frame", depthColorMat);
            Cv2.WaitKey(1);

            // Dispose Mats
            //cvMat.Dispose();
            depthColorMat.Dispose();
        }
    }

    static Mat RenderDepthFrame(short[] depthData, int width, int height)
    {
        // Convert depth data to millimeters and then to 8-bit grayscale
        ushort[] depthInMm = new ushort[depthData.Length];
        for (int i = 0; i < depthData.Length; i++)
        {
            depthInMm[i] = (ushort)(depthData[i] >> 3); // Extract depth in mm
        }

        Mat depthMat16U = new Mat(height, width, MatType.CV_16UC1, depthInMm);
        Mat depthMat8U = new Mat();
        double maxDepth = 4000; // Adjust based on your max expected depth
        depthMat16U.ConvertTo(depthMat8U, MatType.CV_8UC1, 255.0 / maxDepth);
        Cv2.Subtract(255, depthMat8U, depthMat8U); // Invert for better visualization

        // Convert to BGR for color annotations
        Mat depthColorMat = new Mat();
        Cv2.CvtColor(depthMat8U, depthColorMat, ColorConversionCodes.GRAY2BGR);

        return depthColorMat;
    }
}

public class Detection
{
    public string Label { get; set; }
    public float Confidence { get; set; }
    public int[] BBox { get; set; }  // [x1, y1, x2, y2]
}