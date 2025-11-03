using SQLite;
using WildLensApp.Models;

namespace WildLensApp.Services;

public class HistoryService
{
    private readonly SQLiteAsyncConnection _db;

    public HistoryService()
    {
        var dbPath = Path.Combine(FileSystem.AppDataDirectory, "wildlens.db3");
        _db = new SQLiteAsyncConnection(dbPath);
        _ = _db.CreateTableAsync<DetectionHistoryItem>();
    }

    public Task<int> AddAsync(DetectionHistoryItem item) => _db.InsertAsync(item);
    public Task<List<DetectionHistoryItem>> GetAllAsync() => _db.Table<DetectionHistoryItem>().OrderByDescending(x => x.Timestamp).ToListAsync();
}
