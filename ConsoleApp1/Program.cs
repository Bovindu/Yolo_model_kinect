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

    static void Main()
    {
        // Initialize ZeroMQ (connect to Python YOLO server)
        _zmqSocket = new RequestSocket();
        _zmqSocket.Connect("tcp://localhost:5555");

        // Initialize Kinect
        _sensor = KinectSensor.KinectSensors.FirstOrDefault(s => s.Status == KinectStatus.Connected);
        if (_sensor != null)
        {
            _sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
            _sensor.ColorFrameReady += Sensor_ColorFrameReady;
            _sensor.Start();
            Console.WriteLine("Kinect streaming started. Press Enter to exit.");
            Console.ReadLine();

            // Cleanup
            _sensor.Stop();
            _zmqSocket.Dispose();
            Cv2.DestroyAllWindows();
        }
        else
        {
            Console.WriteLine("No Kinect found!");
        }
    }

    static void Sensor_ColorFrameReady(object sender, ColorImageFrameReadyEventArgs e)
    {
        using (var frame = e.OpenColorImageFrame())
        {
            if (frame == null) return;

            // Convert Kinect frame to Bitmap
            byte[] pixelData = new byte[frame.PixelDataLength];
            frame.CopyPixelDataTo(pixelData);
            Bitmap bmp = new Bitmap(frame.Width, frame.Height, PixelFormat.Format32bppRgb);
            BitmapData bmpData = bmp.LockBits(
                new Rectangle(0, 0, bmp.Width, bmp.Height),
                ImageLockMode.WriteOnly,
                bmp.PixelFormat
            );
            Marshal.Copy(pixelData, 0, bmpData.Scan0, pixelData.Length);
            bmp.UnlockBits(bmpData);

            // Convert to OpenCV Mat
            Mat cvMat = BitmapConverter.ToMat(bmp);

            // Send to Python YOLO server
            byte[] jpegBytes;
            using (MemoryStream ms = new MemoryStream())
            {
                bmp.Save(ms, ImageFormat.Jpeg);
                jpegBytes = ms.ToArray();
            }
            _zmqSocket.SendFrame(jpegBytes);

            // Get detections from Python
            string jsonResponse = _zmqSocket.ReceiveFrameString();
            var detections = JsonConvert.DeserializeObject<List<Detection>>(jsonResponse);

            // Draw bounding boxes
            foreach (var d in detections)
            {
                Cv2.Rectangle(
                    cvMat,
                    new OpenCvSharp.Point(d.BBox[0], d.BBox[1]),
                    new OpenCvSharp.Point(d.BBox[2], d.BBox[3]),
                    Scalar.Red,
                    2
                );

                var centerX = (d.BBox[0] + d.BBox[2]) / 2;
                var centerY = (d.BBox[1] + d.BBox[3]) / 2;

                Cv2.Circle(
                    cvMat,
                    new OpenCvSharp.Point(centerX,centerY),
                    5,
                    Scalar.Blue,
                    -1
                );
                Cv2.PutText(
                    cvMat,
                    $"{d.Label} ({d.Confidence:P0})",
                    new OpenCvSharp.Point(d.BBox[0], d.BBox[1] - 10),
                    HersheyFonts.HersheySimplex,
                    0.6,
                    Scalar.Green,
                    1
                );
            }

            // Display
            Cv2.ImShow("Kinect + YOLO", cvMat);
            Cv2.WaitKey(1); // Refresh window
        }
    }
}

public class Detection
{
    public string Label { get; set; }
    public float Confidence { get; set; }
    public int[] BBox { get; set; }  // [x1, y1, x2, y2]
}