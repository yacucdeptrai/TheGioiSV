using Microsoft.Maui.Storage;
using Microsoft.Maui.Graphics;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using WildLensApp.Models;
using System.Runtime.InteropServices;

namespace WildLensApp.Services;

public class OnnxInferenceService : IAsyncDisposable
{
    private InferenceSession? _session;
    private string[] _labels = Array.Empty<string>();
    private readonly SemaphoreSlim _initLock = new(1, 1);

    public bool IsLoaded => _session is not null;
    public int InputSize { get; private set; } = 640; // default, will adapt to model input if available

    public async Task InitializeAsync()
    {
        if (IsLoaded) return;
        await _initLock.WaitAsync();
        try
        {
            if (IsLoaded) return;

            // Load model from bundled assets
            using var modelStream = await FileSystem.OpenAppPackageFileAsync("Resources/Assets/model.onnx");
            using var ms = new MemoryStream();
            await modelStream.CopyToAsync(ms);
            var opts = new SessionOptions();
            // Use CPU by default; mobile GPU EPs require extra setup
            opts.AppendExecutionProvider_CPU();
            _session = new InferenceSession(ms.ToArray(), opts);

            // Load labels
            using var labelsStream = await FileSystem.OpenAppPackageFileAsync("Resources/Assets/labels.txt");
            using var sr = new StreamReader(labelsStream, Encoding.UTF8);
            var list = new List<string>();
            while (!sr.EndOfStream)
            {
                var line = (await sr.ReadLineAsync())?.Trim();
                if (!string.IsNullOrWhiteSpace(line)) list.Add(line!);
            }
            _labels = list.ToArray();

            // Derive input size if possible
            var input = _session.InputMetadata.First().Value.Dimensions;
            // Expect [1,3,H,W] or [N,3,H,W]
            if (input.Length == 4 && input[2] > 0 && input[3] > 0)
            {
                InputSize = Math.Min(input[2], input[3]);
            }
        }
        finally
        {
            _initLock.Release();
        }
    }

    public async Task<List<Detection>> DetectAsync(byte[] rgbaBytes, int width, int height, CancellationToken ct = default)
    {
        if (!IsLoaded) await InitializeAsync();
        if (_session is null) throw new InvalidOperationException("ONNX session not initialized");

        // Preprocess RGBA -> NCHW FP32 normalized 0..1, letterbox to square InputSize
        var (tensor, scaleX, scaleY, padX, padY) = Preprocess(rgbaBytes, width, height, InputSize);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_session.InputMetadata.First().Key, tensor)
        };

        using var results = _session.Run(inputs);
        // YOLOv8 ONNX usually outputs a single [1,84,8400] tensor (for COCO 80 classes). For custom, dims vary.
        // We'll attempt to handle common case where output is [1, N, anchors] or [1, anchors, N].
        var firstOut = results.First().AsEnumerable<float>().ToArray();
        var outMeta = _session.OutputMetadata.First().Value;
        var dims = outMeta.Dimensions.ToArray();

        // Try to reshape heuristically
        int numAnchors, numAttrs;
        if (dims.Length == 3)
        {
            // [1, num_attrs, num_anchors] or [1, num_anchors, num_attrs]
            if (dims[1] < dims[2]) { numAttrs = dims[1]; numAnchors = dims[2]; }
            else { numAttrs = dims[2]; numAnchors = dims[1]; }
        }
        else if (dims.Length == 2)
        {
            // [num_anchors, num_attrs]
            numAnchors = dims[0];
            numAttrs = dims[1];
        }
        else
        {
            // Fallback guess
            numAnchors = firstOut.Length / 84;
            numAttrs = 84;
        }

        var boxes = Postprocess(firstOut, numAnchors, numAttrs, scaleX, scaleY, padX, padY, InputSize, _labels.Length);
        return boxes;
    }

    private static (DenseTensor<float> tensor, float scaleX, float scaleY, float padX, float padY) Preprocess(byte[] rgba, int w, int h, int input)
    {
        // Letterbox to input x input, keeping aspect ratio
        float scale = Math.Min((float)input / w, (float)input / h);
        int newW = (int)Math.Round(w * scale);
        int newH = (int)Math.Round(h * scale);
        int padW = input - newW;
        int padH = input - newH;
        int padLeft = padW / 2;
        int padTop = padH / 2;

        var tensor = new DenseTensor<float>(new[] { 1, 3, input, input });

        // Simple nearest resize
        for (int y = 0; y < input; y++)
        {
            int srcY = Math.Clamp((int)Math.Floor((y - padTop) / scale), 0, h - 1);
            bool yIn = y >= padTop && y < padTop + newH;
            for (int x = 0; x < input; x++)
            {
                int srcX = Math.Clamp((int)Math.Floor((x - padLeft) / scale), 0, w - 1);
                bool xIn = x >= padLeft && x < padLeft + newW;

                int dstIdx = y * input + x;
                if (xIn && yIn)
                {
                    int srcIdx = (srcY * w + srcX) * 4; // RGBA
                    float r = rgba[srcIdx] / 255f;
                    float g = rgba[srcIdx + 1] / 255f;
                    float b = rgba[srcIdx + 2] / 255f;
                    tensor[0, 0, y, x] = r;
                    tensor[0, 1, y, x] = g;
                    tensor[0, 2, y, x] = b;
                }
                else
                {
                    tensor[0, 0, y, x] = 0;
                    tensor[0, 1, y, x] = 0;
                    tensor[0, 2, y, x] = 0;
                }
            }
        }

        return (tensor, (float)newW / input, (float)newH / input, (float)padLeft / input, (float)padTop / input);
    }

    private List<Detection> Postprocess(float[] output, int numAnchors, int numAttrs, float scaleX, float scaleY, float padX, float padY, int input, int numClasses)
    {
        // Assume YOLOv8: attrs = 4 box + 1 obj + numClasses
        int clsCount = Math.Max(1, numAttrs - 5);
        var dets = new List<(Detection det, float score)>();

        for (int a = 0; a < numAnchors; a++)
        {
            int baseIdx = a * numAttrs;
            // Determine layout: Sometimes it's [attrs, anchors]. We'll try both
            if (output.Length == numAnchors * numAttrs)
            {
                // treat as [anchors, attrs]
            }

            float x = output[baseIdx + 0];
            float y = output[baseIdx + 1];
            float w = output[baseIdx + 2];
            float h = output[baseIdx + 3];
            float obj = output[baseIdx + 4];

            // Convert from center x,y,w,h normalized to [0..1] if already, else assume normalized
            float conf;
            int bestClass = 0;
            float bestScore = 0f;
            for (int c = 0; c < Math.Min(clsCount, _labels.Length); c++)
            {
                float clsScore = output[baseIdx + 5 + c];
                if (clsScore > bestScore)
                {
                    bestScore = clsScore;
                    bestClass = c;
                }
            }
            conf = obj * bestScore;
            if (conf < 0.4f) continue;

            // Map back to letterboxed space and then to original normalized coords
            // Here we assume x,y are center in [0,1] of the input image
            float x0 = (x - w / 2f);
            float y0 = (y - h / 2f);
            float x1 = (x + w / 2f);
            float y1 = (y + h / 2f);

            // Remove padding scaling
            x0 = Math.Clamp((x0 - padX) / scaleX, 0, 1);
            y0 = Math.Clamp((y0 - padY) / scaleY, 0, 1);
            x1 = Math.Clamp((x1 - padX) / scaleX, 0, 1);
            y1 = Math.Clamp((y1 - padY) / scaleY, 0, 1);

            var rect = new RectF(x0, y0, Math.Max(0, x1 - x0), Math.Max(0, y1 - y0));
            var det = new Detection
            {
                Label = bestClass < _labels.Length ? _labels[bestClass] : $"cls_{bestClass}",
                Confidence = conf,
                Box = rect
            };
            dets.Add((det, conf));
        }

        // NMS
        var results = Nms(dets.Select(d => d.det).ToList(), 0.5f);
        return results;
    }

    private static List<Detection> Nms(List<Detection> dets, float iouThresh)
    {
        var result = new List<Detection>();
        var sorted = dets.OrderByDescending(d => d.Confidence).ToList();
        while (sorted.Count > 0)
        {
            var best = sorted[0];
            result.Add(best);
            sorted.RemoveAt(0);
            sorted = sorted.Where(d => IoU(best.Box, d.Box) < iouThresh).ToList();
        }
        return result;
    }

    private static float IoU(RectF a, RectF b)
    {
        float interX0 = Math.Max(a.Left, b.Left);
        float interY0 = Math.Max(a.Top, b.Top);
        float interX1 = Math.Min(a.Right, b.Right);
        float interY1 = Math.Min(a.Bottom, b.Bottom);
        float inter = Math.Max(0, interX1 - interX0) * Math.Max(0, interY1 - interY0);
        float union = a.Width * a.Height + b.Width * b.Height - inter;
        return union <= 0 ? 0 : inter / union;
    }

    public ValueTask DisposeAsync()
    {
        _session?.Dispose();
        _session = null;
        return ValueTask.CompletedTask;
    }
}
