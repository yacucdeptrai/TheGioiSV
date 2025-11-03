using Microsoft.Maui.Controls;
using WildLensApp.Models;
using WildLensApp.ViewModels;

namespace WildLensApp.Pages;

public partial class DetailsPage : ContentPage, IQueryAttributable
{
    private DetailsViewModel? _vm;

    public DetailsPage()
    {
        InitializeComponent();
    }

    protected override void OnHandlerChanged()
    {
        base.OnHandlerChanged();
        if (_vm is null && Handler?.MauiContext is not null)
        {
            _vm = Handler.MauiContext.Services.GetService(typeof(DetailsViewModel)) as DetailsViewModel;
            BindingContext = _vm;
        }
    }

    public void ApplyQueryAttributes(IDictionary<string, object> query)
    {
        if (_vm is null) return;
        if (query.TryGetValue("det", out var obj) && obj is Detection det)
        {
            _vm.Load(det);
        }
    }
}
