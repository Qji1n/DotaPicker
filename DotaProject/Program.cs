using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace DotaProject
{
    public class Hero
    {
        public int HeroId { get; set; }
    }

    internal class Program
    {
        public class JsonMatchData
        {
            public long MatchId { get; set; }
            public List<PickBan> PicksBans { get; set; }
            public bool RadiantWin { get; set; }
        }

        public class PickBan
        {
            public bool is_pick { get; set; }
            public int hero_id { get; set; }
            public int team { get; set; }
        }

        public static List<DotaMatchData> LoadMatchesFromJsonFiles(string folderPath)
        {
            var matchesData = new List<DotaMatchData>();

            // Получить все файлы JSON из папки
            var jsonFiles = Directory.GetFiles(folderPath, "*.json");

            foreach (var file in jsonFiles)
            {
                Console.WriteLine(file);
                var jsonData = File.ReadAllText(file);
                var matchesDict = JsonSerializer.Deserialize<Dictionary<string, JsonMatchData>>(jsonData);
                var matchesList = matchesDict.Values.ToList(); // если вам нужен список матчей

                var trainingData = matchesList.Select(match =>
                {
                    var radiantHeroes = match.PicksBans.Where(p => p.team == 0 && p.is_pick).Select(p => (float)p.hero_id).ToList();
                    var direHeroes = match.PicksBans.Where(p => p.team == 1 && p.is_pick).Select(p => (float)p.hero_id).ToList();

                    return new DotaMatchData
                    {
                        RadiantFeatures = ModelTrainer.ConvertToFeatures(radiantHeroes),
                        DireFeatures = ModelTrainer.ConvertToFeatures(direHeroes),
                        RadiantWin = match.RadiantWin
                    };
                }).ToList();

                // Добавьте trainingData в matchesData
                matchesData.AddRange(trainingData);
            }

            return matchesData;
        }


        public static async Task Main(string[] args)
        {
            //создаем объект
            var stopwatch = new Stopwatch();
            //засекаем время начала операции
            stopwatch.Start();
            var processor = new StatisticsProcessor();
            var proccesorMatchID = new MatchIDsProcessor();
            var matchesId = await proccesorMatchID.GetMatchesIdAsync(5000);
            Console.WriteLine($"matchesId {matchesId.Count}");
            //останавливаем счётчик
            stopwatch.Stop();
            //смотрим сколько миллисекунд было затрачено на выполнение
            Console.WriteLine($"TIME FOR GetMatchesIdAsync {stopwatch.ElapsedMilliseconds/1000} secs");
            //создаем объект
            stopwatch = new Stopwatch();
            //засекаем время начала операции
            stopwatch.Start();
            var data = await processor.GetMatchesByMatchIdAsync(matchesId);
            //останавливаем счётчик
            stopwatch.Stop();
            //смотрим сколько миллисекунд было затрачено на выполнение
            Console.WriteLine($"TIME FOR GetMatchesByMatchIdAsync {stopwatch.ElapsedMilliseconds / 1000} secs");
            //var data = await processor.GetMatchesInRangeAsync(long.Parse("7430000000"), long.Parse("7430000100"));
            //var trainingData = data.Where(match => match != null).Select(match => new DotaMatchData
            //{
            //    RadiantFeatures = ModelTrainer.ConvertToFeatures(match.RadiantFeatures),
            //    DireFeatures = ModelTrainer.ConvertToFeatures(match.DireFeatures),
            //    RadiantWin = match.RadiantWin,
            //});
            



            //var modelPath = "C:/Users/Vladimir/Desktop/DotaModel/model.zip";
            //var trainer = new ModelTrainer(modelPath);
            //var trainingData = LoadMatchesFromJsonFiles("dataset");
            //Console.WriteLine(trainingData.Count);
            //trainer.TrainAndSaveModel(trainingData);

            //var predictor = new DotaPredictor(modelPath);
            //var testData = new DotaMatchData
            //{
            //    RadiantFeatures = ModelTrainer.ConvertToFeatures(new float[] { 1, 2, 3, 4, 5 }),
            //    DireFeatures = ModelTrainer.ConvertToFeatures(new float[] { 6, 7, 8, 9, 10 })
            //};

            //var prediction = predictor.Predict(testData);
            //var isRadiantWin = prediction.IsRadiantWin;
            //var winProbability = prediction.WinProbability;
            //Console.WriteLine(isRadiantWin ? $"Radiant is predicted to win with a probability of {winProbability * 100}%!" : $"Dire is predicted to win with a probability of {100 - winProbability * 100}%!");

        }

    }

    public class DotaMatchData
    {
        [VectorType(124)]
        public float[] RadiantFeatures { get; set; }

        [VectorType(124)]
        public float[] DireFeatures { get; set; }

        [ColumnName("Label")]
        public bool RadiantWin { get; set; }

    }

    public class ModelTrainer
    {
        private readonly MLContext _context;
        private string _modelPath;

        public ModelTrainer(string modelPath)
        {
            _context = new MLContext();
            _modelPath = modelPath;
        }

        public void TrainAndSaveModel(IEnumerable<DotaMatchData> matchesData)
        {
            var data = _context.Data.LoadFromEnumerable(matchesData);

            var pipeline = _context.Transforms.Concatenate("Features", "RadiantFeatures", "DireFeatures")
                .Append(_context.Transforms.NormalizeMinMax("Features"))
                .Append(_context.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            var model = pipeline.Fit(data);
            _context.Model.Save(model, data.Schema, _modelPath);
        }

        public static float[] ConvertToFeatures(IEnumerable<float> heroIds)
        {
            float[] features = new float[124];
            foreach (var id in heroIds)
            {
                var new_id = id;
                if (id == 126)
                    new_id = 0;
                else if (id == 128)
                    new_id = 24;
                else if (id == 129)
                    new_id = 115;
                else if (id == 135)
                    new_id = 116;
                else if (id == 136)
                    new_id = 117;
                else if (id == 137)
                    new_id = 118;
                else if (id == 138)
                    new_id = 122;

                features[(int)new_id] = 1;
            }
            return features;
        }
    }

    public class DotaPredictor
    {
        private readonly MLContext _context;
        private ITransformer _model;

        public DotaPredictor(string modelPath)
        {
            _context = new MLContext();
            _model = _context.Model.Load(modelPath, out var modelSchema);
        }

        public DotaMatchPrediction Predict(DotaMatchData sampleMatch)
        {
            var predictionEngine = _context.Model.CreatePredictionEngine<DotaMatchData, DotaMatchPrediction>(_model);
            return predictionEngine.Predict(sampleMatch);
        }
    }

    public class DotaMatchPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsRadiantWin;

        [ColumnName("Probability")]
        public float WinProbability;
    }


}
