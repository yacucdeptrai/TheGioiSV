using Microsoft.Maui.Controls;
using WildLensApp.ViewModels;

namespace WildLensApp.Pages;

public partial class HistoryPage : ContentPage
{
    private HistoryViewModel? _vm;

    public HistoryPage()
    {
        InitializeComponent();
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        if (_vm is null && Handler?.MauiContext is not null)
        {
            _vm = Handler.MauiContext.Services.GetService(typeof(HistoryViewModel)) as HistoryViewModel;
            BindingContext = _vm;
        }
        if (_vm is not null)
        {
            await _vm.LoadAsync();
        }
    }

    private async void OnRefreshClicked(object? sender, EventArgs e)
    {
        if (_vm is not null)
            await _vm.LoadAsync();
    }
}
