using Microsoft.Maui.Controls;

namespace WildLensApp;

public partial class AppShell : Shell
{
    public AppShell()
    {
        InitializeComponent();
        Routing.RegisterRoute("Details", typeof(Pages.DetailsPage));
    }
}
