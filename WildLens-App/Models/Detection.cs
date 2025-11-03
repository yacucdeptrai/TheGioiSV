using Microsoft.Maui.Graphics;

namespace WildLensApp.Models;

public class Detection
{
    public string Label { get; set; } = string.Empty;
    public float Confidence { get; set; }
    // Bounding box normalized [0..1] relative to image width/height
    public RectF Box { get; set; }
}
