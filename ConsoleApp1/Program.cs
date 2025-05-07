using System.Collections.Generic;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Kinect;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System.Linq;

class Program
{
    static KinectSensor _sensor;
    static RequestSocket _zmqSocket;

    static void Main()
    {
        // Initialize ZeroMQ (connect to Python)
        _zmqSocket = new RequestSocket();
        _zmqSocket.Connect("tcp://localhost:5555");

        // Initialize Kinect
        _sensor = KinectSensor.KinectSensors.FirstOrDefault();
        if (_sensor != null)
        {
            _sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
            _sensor.ColorFrameReady += Sensor_ColorFrameReady;
            _sensor.Start();
            Console.WriteLine("Kinect running. Press Enter to exit.");
            Console.ReadLine();
            _sensor.Stop();
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
            BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height),
                                            ImageLockMode.WriteOnly, bmp.PixelFormat);
            Marshal.Copy(pixelData, 0, bmpData.Scan0, pixelData.Length);
            bmp.UnlockBits(bmpData);

            // Convert to JPEG bytes
            byte[] jpegBytes;
            using (MemoryStream ms = new MemoryStream())
            {
                bmp.Save(ms, ImageFormat.Jpeg);
                jpegBytes = ms.ToArray();
            }

            // Send to Python and get detections
            _zmqSocket.SendFrame(jpegBytes);
            string jsonResponse = _zmqSocket.ReceiveFrameString();
            var detections = JsonConvert.DeserializeObject<List<Detection>>(jsonResponse);

            // Display detections in console
            Console.Clear();
            foreach (var d in detections)
            {
                Console.WriteLine($"{d.Label} ({d.Confidence:P0}) at [X:{d.BBox[0]}, Y:{d.BBox[1]}]");
            }
        }
    }
}

public class Detection
{
    public string Label { get; set; }
    public float Confidence { get; set; }
    public int[] BBox { get; set; }  // Keep as int[]
}