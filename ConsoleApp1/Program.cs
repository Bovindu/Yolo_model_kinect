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

            // Convert to bitmap
            Bitmap bmp = new Bitmap(colorFrame.Width, colorFrame.Height, PixelFormat.Format32bppRgb);
            BitmapData bmpData = bmp.LockBits(
                new Rectangle(0, 0, bmp.Width, bmp.Height),
                ImageLockMode.WriteOnly,
                bmp.PixelFormat
            );
            Marshal.Copy(_colorData, 0, bmpData.Scan0, _colorData.Length);
            bmp.UnlockBits(bmpData);

            // Convert to OpenCV Mat
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

            foreach (var d in detections)
            {
                // Existing drawing code
                var centerX = (d.BBox[0] + d.BBox[2]) / 2;
                var centerY = (d.BBox[1] + d.BBox[3]) / 2;

                // Get depth at color coordinates
                int depthX = (int)(centerX * _depthWidth / colorFrame.Width);
                int depthY = (int)(centerY * _depthHeight / colorFrame.Height);
                int depthIndex = depthY * _depthWidth + depthX;

                if (depthIndex >= 0 && depthIndex < _depthData.Length)
                {
                    short depthValue = _depthData[depthIndex];
                    int depthInMm = depthValue >> DepthImageFrame.PlayerIndexBitmaskWidth;
                                       
                    Cv2.Circle(cvMat, new OpenCvSharp.Point((d.BBox[0] + d.BBox[2]) / 2, (d.BBox[1] + d.BBox[3]) / 2), 5, Scalar.Red, -1);

                    Cv2.PutText(
                        cvMat,
                        $"{depthInMm}mm",
                        new OpenCvSharp.Point(d.BBox[0], d.BBox[1] - 30),
                        HersheyFonts.HersheySimplex,
                        0.6,
                        Scalar.White,
                        1
                    );
                }
            }

            Cv2.ImShow("Kinect + YOLO", cvMat);
            Cv2.WaitKey(1);
        }
    }
}

public class Detection
{
    public string Label { get; set; }
    public float Confidence { get; set; }
    public int[] BBox { get; set; }  // [x1, y1, x2, y2]
}