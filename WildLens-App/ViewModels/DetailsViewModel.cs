using CommunityToolkit.Mvvm.ComponentModel;
using WildLensApp.Models;

namespace WildLensApp.ViewModels;

public partial class DetailsViewModel : ObservableObject
{
    [ObservableProperty]
    private Detection? detection;

    public void Load(Detection det)
    {
        Detection = det;
    }
}
