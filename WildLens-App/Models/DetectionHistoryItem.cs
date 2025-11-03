using SQLite;
using System;

namespace WildLensApp.Models;

public class DetectionHistoryItem
{
    [PrimaryKey, AutoIncrement]
    public int Id { get; set; }

    public string Label { get; set; } = string.Empty;
    public float Confidence { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string? ImagePath { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public float Width { get; set; }
    public float Height { get; set; }
}