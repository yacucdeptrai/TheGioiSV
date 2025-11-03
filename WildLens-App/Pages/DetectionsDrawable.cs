using Microsoft.Maui.Graphics;
using WildLensApp.ViewModels;

namespace WildLensApp.Pages;

public class DetectionsDrawable : IDrawable
{
    private readonly ViewModels.CameraViewModel _vm;
    public DetectionsDrawable(ViewModels.CameraViewModel vm)
    {
        _vm = vm;
    }

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        canvas.SaveState();
        canvas.StrokeColor = Colors.LimeGreen;
        canvas.StrokeSize = 2f;
        canvas.FontColor = Colors.Yellow;
        canvas.FontSize = 14;

        foreach (var d in _vm.Detections)
        {
            var r = d.Box;
            var rect = new RectF(r.X * (float)dirtyRect.Width,
                                 r.Y * (float)dirtyRect.Height,
                                 r.Width * (float)dirtyRect.Width,
                                 r.Height * (float)dirtyRect.Height);

            canvas.DrawRectangle(rect);
            var label = $"{d.Label} {(d.Confidence):P0}";
            canvas.FillColor = new Color(0,0,0,0.5f);
            var labelRect = new RectF(rect.X, rect.Y - 18, rect.Width, 18);
            canvas.FillRectangle(labelRect);
            canvas.DrawString(label, rect.X + 4, rect.Y - 16, HorizontalAlignment.Left);
        }

        // Highlight selected
        if (_vm.SelectedDetection is not null)
        {
            var r = _vm.SelectedDetection.Box;
            var rect = new RectF(r.X * (float)dirtyRect.Width,
                                 r.Y * (float)dirtyRect.Height,
                                 r.Width * (float)dirtyRect.Width,
                                 r.Height * (float)dirtyRect.Height);
            canvas.StrokeColor = Colors.OrangeRed;
            canvas.StrokeSize = 3f;
            canvas.DrawRectangle(rect);
        }

        canvas.RestoreState();
    }
}
