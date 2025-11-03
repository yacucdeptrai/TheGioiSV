using Microsoft.Maui;
using Microsoft.Maui.Controls;

namespace WildLensApp;

public partial class App : Application
{
    public App()
    {
        InitializeComponent();
        MainPage = new AppShell();
    }
}
