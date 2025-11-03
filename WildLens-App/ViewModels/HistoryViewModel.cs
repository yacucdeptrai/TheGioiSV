using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using WildLensApp.Models;
using WildLensApp.Services;

namespace WildLensApp.ViewModels;

public partial class HistoryViewModel : ObservableObject
{
    private readonly HistoryService _history;

    public ObservableCollection<DetectionHistoryItem> Items { get; } = new();

    public HistoryViewModel(HistoryService history)
    {
        _history = history;
    }

    public async Task LoadAsync()
    {
        var list = await _history.GetAllAsync();
        MainThread.BeginInvokeOnMainThread(() =>
        {
            Items.Clear();
            foreach (var it in list) Items.Add(it);
        });
    }
}
