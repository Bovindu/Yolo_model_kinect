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
    static DepthImagePoint[] _mappedDepthPoints = new DepthImagePoint[640 * 480];


    static void Main()
    {
        _zmqSocket = new RequestSocket();
        _zmqSocket.Connect("tcp://localhost:5555");

        _sensor = KinectSensor.KinectSensors.FirstOrDefault();
        if (_sensor != null)
        {
            // Enable streams
            _sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
            _sensor.DepthStream.Enable(DepthImageFormat.Resolution320x240Fps30); ;

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

            

            foreach (var d in detections)
            {
                var centerX = (d.BBox[0] + d.BBox[2]) / 2;
                var centerY = (d.BBox[1] + d.BBox[3]) / 2;

                int colorX = (int)centerX;
                int colorY = (int)centerY;
                int colorIndex = colorY * 640 + colorX;

                // Convert short[] to DepthImagePixel[] (do this once per frame)
                DepthImagePixel[] depthPixels = new DepthImagePixel[320 * 240];
                for (int i = 0; i < depthPixels.Length; i++)
                {
                    depthPixels[i].Depth = (short)(_depthData[i] >> DepthImageFrame.PlayerIndexBitmaskWidth);
                }

                // Map color pixel to depth pixel
                _sensor.CoordinateMapper.MapColorFrameToDepthFrame(
                    ColorImageFormat.RgbResolution640x480Fps30,
                    DepthImageFormat.Resolution320x240Fps30,
                    depthPixels,
                    _mappedDepthPoints
                );


                if (colorIndex >= 0 && colorIndex < _mappedDepthPoints.Length)
                {
                    DepthImagePoint depthPoint = _mappedDepthPoints[colorIndex];
                    int depthInMM = depthPoint.Depth;

                    if (depthInMM > 0)
                    {
                        // Get 3D coordinates in camera space (meters)
                        SkeletonPoint realWorldPoint = _sensor.CoordinateMapper.MapDepthPointToSkeletonPoint(
                            DepthImageFormat.Resolution320x240Fps30,
                            depthPoint
                        );

                        double x = realWorldPoint.X * 1000;
                        double y = realWorldPoint.Y * 1000;
                        double z = realWorldPoint.Z * 1000;

                        // Show on screen
                        string coordText = $"X: {x:F2} Y: {y:F2} Z: {z:F2}";
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
                }


            }

            // Show frames
            Cv2.ImShow("RGB Frame", cvMat);            
            Cv2.WaitKey(1);

            // Dispose Mats
            cvMat.Dispose();            
        }
    }

    
}

public class Detection
{
    public string Label { get; set; }
    public float Confidence { get; set; }
    public int[] BBox { get; set; }  // [x1, y1, x2, y2]
}
