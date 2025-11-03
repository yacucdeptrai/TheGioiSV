using CommunityToolkit.Maui;
using Microsoft.Extensions.Logging;
using Microsoft.Maui;
using Microsoft.Maui.Controls.Hosting;
using Microsoft.Maui.Hosting;
using WildLensApp.Services;
using WildLensApp.ViewModels;

namespace WildLensApp;

public static class MauiProgram
{
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder
            .UseMauiApp<App>()
            .UseMauiCommunityToolkit()
            .ConfigureFonts(fonts =>
            {
                fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
            });

#if DEBUG
        builder.Logging.AddDebug();
#endif

        // Services
        builder.Services.AddSingleton<OnnxInferenceService>();
        builder.Services.AddSingleton<HistoryService>();

        // ViewModels
        builder.Services.AddTransient<ViewModels.CameraViewModel>();
        builder.Services.AddTransient<ViewModels.DetailsViewModel>();
        builder.Services.AddTransient<ViewModels.HistoryViewModel>();

        // Pages
        builder.Services.AddTransient<Pages.CameraPage>();
        builder.Services.AddTransient<Pages.DetailsPage>();
        builder.Services.AddTransient<Pages.HistoryPage>();

        return builder.Build();
    }
}
