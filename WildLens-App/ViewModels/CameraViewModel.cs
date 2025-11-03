using System.Collections.ObjectModel;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.Graphics;
using WildLensApp.Models;
using WildLensApp.Services;

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

        // Load image to get RGBA buffer
        var image = Microsoft.Maui.Graphics.Platform.PlatformImage.FromStream(new MemoryStream(imageBytes));
        int width = (int)image.Width;
        int height = (int)image.Height;
        var rgba = image.AsImageSource().ToByteArray(); // fallback; platform-specific; may return null
        if (rgba is null)
        {
            // Fallback: create via Skia
            using var skBitmap = new SkiaSharp.SKBitmap();
            using var skStream = new SkiaSharp.SKMemoryStream(imageBytes);
            if (SkiaSharp.SKBitmap.Decode(skStream) is { } bmp)
            {
                rgba = bmp.Bytes.ToArray();
                width = bmp.Width;
                height = bmp.Height;
            }
            else return;
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
