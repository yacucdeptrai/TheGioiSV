using System.Collections.ObjectModel;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui;
using Microsoft.Maui.ApplicationModel;
using WildLensApp.Models;
using WildLensApp.Services;
using SkiaSharp;

namespace WildLensApp.ViewModels;

public partial class CameraViewModel : ObservableObject
{
    private readonly OnnxInferenceService _inference;
    private readonly HistoryService _history;

    [ObservableProperty]
    private Detection? selectedDetection;

    public ObservableCollection<Detection> Detections { get; } = new();

    public CameraViewModel(OnnxInferenceService inference, HistoryService history)
    {
        _inference = inference;
        _history = history;
    }

    [RelayCommand]
    public async Task AnalyzeImageAsync(FileResult file)
    {
        if (file is null) return;
        using var stream = await file.OpenReadAsync();
        using var ms = new MemoryStream();
        await stream.CopyToAsync(ms);
        var imageBytes = ms.ToArray();

        // Decode via SkiaSharp and extract RGBA32 pixel buffer
        int width;
        int height;
        byte[] rgba;
        using (var skStream = new SKMemoryStream(imageBytes))
        using (var bmp = SKBitmap.Decode(skStream))
        {
            if (bmp == null) return;
            width = bmp.Width;
            height = bmp.Height;

            // Ensure format is RGBA8888
            if (bmp.ColorType != SKColorType.Rgba8888)
            {
                using var converted = new SKBitmap(width, height, SKColorType.Rgba8888, SKAlphaType.Premul);
                bmp.CopyTo(converted);
                rgba = converted.Bytes.ToArray();
            }
            else
            {
                rgba = bmp.Bytes.ToArray();
            }
        }

        var dets = await _inference.DetectAsync(rgba, width, height);

        MainThread.BeginInvokeOnMainThread(() =>
        {
            Detections.Clear();
            foreach (var d in dets) Detections.Add(d);
        });
    }

    public void UpdateSelectionFromTap(Point tap, Size canvasSize)
    {
        // Convert tap to normalized
        float nx = (float)(tap.X / canvasSize.Width);
        float ny = (float)(tap.Y / canvasSize.Height);
        // Find the first detection that contains the tap
        var hit = Detections.FirstOrDefault(d => d.Box.Contains(nx, ny));
        SelectedDetection = hit;
    }
}
