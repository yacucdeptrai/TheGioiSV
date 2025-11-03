using CommunityToolkit.Maui.Views;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Storage;
using WildLensApp.ViewModels;

namespace WildLensApp.Pages;

public partial class CameraPage : ContentPage
{
    private CameraViewModel? _vm;

    public CameraPage()
    {
        InitializeComponent();
    }

    protected override void OnHandlerChanged()
    {
        base.OnHandlerChanged();
        if (_vm is null && Handler?.MauiContext is not null)
        {
            _vm = Handler.MauiContext.Services.GetService(typeof(CameraViewModel)) as CameraViewModel;
            BindingContext = _vm;
            if (_vm is not null)
            {
                Overlay.Drawable = new DetectionsDrawable(_vm);
            }
        }
    }

    private async void OnPickImageClicked(object? sender, EventArgs e)
    {
        if (_vm is null) return;
        var result = await FilePicker.Default.PickAsync(new PickOptions
        {
            PickerTitle = "Pick an image"
        });
        if (result != null)
        {
            await _vm.AnalyzeImageAsync(result);
            Overlay.Invalidate();
        }
    }

    private void OnOverlayTapped(object? sender, TappedEventArgs e)
    {
        if (_vm is null || e.GetPosition(Overlay) is not Point p) return;
        _vm.UpdateSelectionFromTap(p, Overlay.Bounds.Size);
        Overlay.Invalidate();
        if (_vm.SelectedDetection is not null)
        {
            Shell.Current.GoToAsync("Details", new Dictionary<string, object>
            {
                { "det", _vm.SelectedDetection }
            });
        }
    }
}
