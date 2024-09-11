//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using System.Net.Http;
//using Newtonsoft.Json;
//using System.Threading.Tasks;
//using System.Dynamic;
//using System.Text.RegularExpressions;
//using System.Net;
//using System.Xml.Serialization;
//using Polly;
//using System.Collections.Concurrent;

//namespace DotaProject
//{
//    public class MatchDataToSave
//    {
//        public long MatchId { get; set; }
//        public IList<PickBan> PicksBans { get; set; }
//        public bool RadiantWin { get; set; }

//        public IList<PlayerPicks> Players { get; set; }

//    }

//    public class ProxyInfo
//    {
//        public string IP { get; set; }
//        public int Port { get; set; }
//        public string Login { get; set; }
//        public string Password { get; set; }
//    }

//    public class OpenDotaApiService
//    {
//        private readonly List<ProxyInfo> ProxyAddresses = new List<ProxyInfo>
//        {
//            new ProxyInfo { IP = "45.89.70.83", Port = 63898, Login = "mrzzb1L3", Password = "UWKxB6zk" },
//            new ProxyInfo { IP = "45.139.127.96", Port = 62470, Login = "mrzzb1L3", Password = "UWKxB6zk" },
//            //new ProxyInfo { IP = "78.47.96.120", Port = 3128 },
//            //new ProxyInfo { IP = "136.243.90.203", Port = 80 }
//        };

//        private ConcurrentDictionary<long, int> matchRetryCounts = new ConcurrentDictionary<long, int>();


//        private static readonly HttpClient SharedHttpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };


//        private ConcurrentBag<MatchDataToSave> matchesToSave = new ConcurrentBag<MatchDataToSave>();


//        private const string BaseUrlOpenDotaAPI = "api.opendota.com/api/";
//        private const string BaseUrl = "http://api.steampowered.com/";

//        private readonly List<string> ApiKeys = new List<string>
//        {
//            "496729413D7324E9765E6049213B01B9", // мейн 2
//            "47E9457E827B6B41AE8381B060B75DE1", // мейн 
//            "566AD70F1E0E637FEFD1E6713060505E", // Виталя
//            "A06EFF59CF282352EF2BB9ABE82C304D", // Вова
//            "0540201D9A7B533F6DFD1D20A39FC890", // мейн 3
//            "ADE73998B5A304780CEC579DC220C5A3", // Ваня
//            "31171B3BE3FF86528CAE2F821A54DFDF",  // Артём
//            "2E15F157F9DC49E59C1D7C5503506241", //5 bot
//            "5E7DB61D6A1D7980DCA2BD90CDC03439", //4 bot
//            "6C6EA695B3412E8FF1919902D8C16613", //3 bot
//            "FD9ECB0735D8E4AFDC16D9D2F6B335CB", //2 bot
//            "A893C825081A1C0EE8F9E7DA5E090928"  //1 bot
//        };
//        private int currentKeyIndex = 0;

//        private ConcurrentBag<long> matchesIdToSave = new ConcurrentBag<long>();
//        private const int minRank = 80;
//        private long maxMatchId = 7505574015L;	

//        private readonly object apiKeyLock = new object();
//        private string GetCurrentApiKey()
//        {
//            lock (apiKeyLock)
//            {
//                string key = ApiKeys[currentKeyIndex];
//                currentKeyIndex = (currentKeyIndex + 1) % ApiKeys.Count;
//                return key;
//            }
//        }

//        public class MatchResponse
//        {
//            public Match result { get; set; }
//        }

//        private readonly SemaphoreSlim matchesSemaphore = new SemaphoreSlim(1, 1);
//        private const int SaveThreshold = 100; // Какое-то пороговое значение
//        public async Task<DotaMatchData> GetMatchAsync(long matchId)
//        {
//            var currentApiKey = GetCurrentApiKey();

//            var retryPolicy = Policy
//                .Handle<HttpRequestException>()
//                .Or<TaskCanceledException>() // Этот обработчик также будет реагировать на истечение времени ожидания
//                .WaitAndRetryAsync(3, retryAttempt =>
//                {
//                    if (!matchRetryCounts.ContainsKey(matchId))
//                        matchRetryCounts[matchId] = 0;

//                    matchRetryCounts[matchId]++;

//                    if (matchRetryCounts[matchId] > 1)
//                        matchId += (long)Math.Pow(2, matchRetryCounts[matchId] - 1); // прибавляем степень двойки

//                    return TimeSpan.FromSeconds(2); // Задержка в 2 секунды перед следующей попыткой
//                },
//                (exception, timeSpan, context) =>
//                {
//                    Console.WriteLine($"Retry after {timeSpan.Seconds} seconds due to: {exception.Message}");
//                });

//            var response2 = await retryPolicy.ExecuteAsync(async () =>
//            {

//                await Task.Delay(1000 / ApiKeys.Count);

//                var response = await SharedHttpClient.GetAsync($"{BaseUrl}IDOTA2Match_570/GetMatchDetails/v1?key={currentApiKey}&match_id={matchId}&");

//                if (!response.IsSuccessStatusCode)
//                {
//                    Console.WriteLine($"Failed to retrieve match data for {matchId}. HTTP Status: {response.StatusCode}");
//                    return null;
//                }
//                var responseBody = await response.Content.ReadAsStringAsync();
//                var matchResponse = JsonConvert.DeserializeObject<MatchResponse>(responseBody);
//                var match = matchResponse.result;

//                Console.WriteLine($"{match.lobby_type}, {match.duration / 60}, {match.skill}");

//                if (match.picks_bans != null)
//                {
//                    matchesToSave.Add(new MatchDataToSave
//                    {
//                        MatchId = matchId,
//                        PicksBans = match.picks_bans,
//                        RadiantWin = match.radiant_win,
//                        Players = match.players
//                    });
//                }

//                await matchesSemaphore.WaitAsync(); // Ждём, пока семафор станет доступен

//                try
//                {
//                    if (matchesToSave.Count >= SaveThreshold)
//                    {
//                        Console.WriteLine("Saving data...");
//                        await SaveMatchesToFileAsync();
//                        matchesToSave.Clear();
//                    }
//                }
//                finally
//                {
//                    matchesSemaphore.Release(); // Освобождаем семафор
//                }

//                return match.ToDotaMatchData();
//            });

//            return null;
//        }

//        //AS
//        public async Task<List<long>> GetMatchIdAsync()
//		{
//            var retryPolicy = Policy
//                .Handle<HttpRequestException>()
//                .Or<TaskCanceledException>() // Этот обработчик также будет реагировать на истечение времени ожидания
//                .WaitAndRetryAsync(3, retryAttempt 
//                    => TimeSpan.FromSeconds(2), // Задержка в 2 секунды перед следующей попыткой
//                (exception, timeSpan, context) =>
//                {
//                    Console.WriteLine($"Retry after {timeSpan.Seconds} seconds due to: {exception.Message}");
//                });

//            Console.WriteLine(1);

//            var response2 = await retryPolicy.ExecuteAsync(async () =>
//            {
//                await Task.Delay(1000);

//                var response = await SharedHttpClient.GetAsync($"https://api.opendota.com/api/publicMatches?" +
//                    $"min_rank={minRank}&less_than_match_id={maxMatchId}");//($"{BaseUrlOpenDotaAPI}publicMatches?min_rank={minRank}");

//                if (!response.IsSuccessStatusCode)
//                {
//                    Console.WriteLine($"Failed to retrieve match data. HTTP Status: {response.StatusCode}");
//                    return null;
//                }

//                var responseBody = await response.Content.ReadAsStringAsync();
//                var matchIds = await ParseMatchIds(responseBody);

//                Console.WriteLine($"matchIds count: {matchIds.Count}");
//                foreach (var matchId in matchIds)
//                {
//                    Console.WriteLine($"matchId: {matchId}");
//                    matchesIdToSave.Add(matchId);
//                }
//                maxMatchId = matchIds[matchIds.Count - 1];
                

//                await matchesSemaphore.WaitAsync(); // Ждём, пока семафор станет доступен

//                try
//                {
//                    Console.WriteLine("Saving matchIds...");
//                    await SaveMatchIdToFileAsync();
//                    matchesIdToSave.Clear();
//                }
//                finally
//                {
//                    matchesSemaphore.Release(); // Освобождаем семафор
//                }

//                return matchIds;
//            });

//            return response2;
//        }

//        //AS
//        private async Task<List<long>> ParseMatchIds(string responseBody)
//        {
//            var indexMatchId = 0;
//            var lastMatchId = 0;
//            var matchesId = new List<long>();
//            while (true)
//            {
//                indexMatchId = responseBody.IndexOf("match_id", indexMatchId) + 10;
//                if (lastMatchId > indexMatchId)
//                    break;
//                if (indexMatchId == -1)
//                    break;

//                await Task.Delay(10);
//                long matchId = 0;
//                while (responseBody[indexMatchId] != ',')
//                {
//                    matchId = 10 * matchId + responseBody[indexMatchId] - '0';
//                    indexMatchId += 1;
//                }
//                matchesId.Add(matchId);
//                lastMatchId = indexMatchId;
//            }
//            return matchesId;
//        }

//        private async Task SaveMatchesToFileAsync()
//        {
//            var datasetDirectory = "dataset/newIDs";
//            if (!Directory.Exists(datasetDirectory))
//                Directory.CreateDirectory(datasetDirectory);


//            var minMatchId = matchesToSave.Min(match => match.MatchId);
//            var maxMatchId = matchesToSave.Max(match => match.MatchId);
//            var filePath = $"{datasetDirectory}/matches_{minMatchId}-{maxMatchId}.json";

//            if (File.Exists(filePath))
//            {
//                using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.None, bufferSize: 4096, useAsync: true))
//                using (var reader = new StreamReader(stream))
//                {
//                    var fileContent = await reader.ReadToEndAsync();
//                    var existingMatches = JsonConvert.DeserializeObject<Dictionary<long, MatchDataToSave>>(fileContent);
//                    foreach (var match in matchesToSave)
//                        existingMatches[match.MatchId] = match;

//                    stream.Seek(0, SeekOrigin.Begin); // Сброс позиции файла на начало
//                    stream.SetLength(0); // Очищаем файл

//                    using (var writer = new StreamWriter(stream))
//                    using (var jsonWriter = new JsonTextWriter(writer))
//                    {
//                        var serializer = new JsonSerializer();
//                        serializer.Formatting = Formatting.Indented;
//                        serializer.Serialize(jsonWriter, existingMatches);
//                        await jsonWriter.FlushAsync();
//                    }
//                }
//            }
//            else
//            {
//                using (var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 4096, useAsync: true))
//                using (var writer = new StreamWriter(stream))
//                using (var jsonWriter = new JsonTextWriter(writer))
//                {
//                    var newMatches = new Dictionary<long, MatchDataToSave>();
//                    foreach (var match in matchesToSave)
//                        newMatches[match.MatchId] = match;

//                    var serializer = new JsonSerializer();
//                    serializer.Formatting = Formatting.Indented;
//                    serializer.Serialize(jsonWriter, newMatches);
//                    await jsonWriter.FlushAsync();
//                }
//            }

//            matchesToSave.Clear();
//        }

//        //AS
//        private async Task SaveMatchIdToFileAsync()
//        {
//            var datasetDirectory = "dataset/matchID";
//            if (!Directory.Exists(datasetDirectory))
//                Directory.CreateDirectory(datasetDirectory);


//            var minMatchId = matchesIdToSave.Min(matchId => matchId);
//            var maxMatchId = matchesIdToSave.Max(matchId => matchId);
//            var filePath = $"{datasetDirectory}/match_IDs_{minMatchId}-{maxMatchId}.json";

//            if (File.Exists(filePath))
//            {
//                using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.None, bufferSize: 4096, useAsync: true))
//                using (var reader = new StreamReader(stream))
//                {
//                    var fileContent = await reader.ReadToEndAsync();
//                    var existingMatchesId = JsonConvert.DeserializeObject<HashSet<long>>(fileContent);
//                    foreach (var matchID in matchesIdToSave)
//                        existingMatchesId.Add(matchID);

//                    stream.Seek(0, SeekOrigin.Begin); // Сброс позиции файла на начало
//                    stream.SetLength(0); // Очищаем файл

//                    using (var writer = new StreamWriter(stream))
//                    using (var jsonWriter = new JsonTextWriter(writer))
//                    {
//                        var serializer = new JsonSerializer();
//                        serializer.Formatting = Formatting.Indented;
//                        serializer.Serialize(jsonWriter, existingMatchesId);
//                        await jsonWriter.FlushAsync();
//                    }
//                }
//            }
//            else
//            {
//                using (var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 4096, useAsync: true))
//                using (var writer = new StreamWriter(stream))
//                using (var jsonWriter = new JsonTextWriter(writer))
//                {
//                    var newMatchesId = new HashSet<long>();
//                    foreach (var matchId in matchesIdToSave)
//                        newMatchesId.Add(matchId);

//                    var serializer = new JsonSerializer();
//                    serializer.Formatting = Formatting.Indented;
//                    serializer.Serialize(jsonWriter, newMatchesId);
//                    await jsonWriter.FlushAsync();
//                }
//            }

//            matchesIdToSave.Clear();
//        }


//        public async Task<List<DotaMatchData>> GetMatchesInRangeAsync(long startMatchId, long endMatchId)
//        {
//            List<DotaMatchData> matches = new List<DotaMatchData>();

//            for (long i = startMatchId; i <= endMatchId; i++)
//            {
//                var match = await GetMatchAsync(i);
//                matches.Add(match);
//            }

//            return matches;
//        }
//    }
//    public class StatisticsProcessor
//    {
//        private static SemaphoreSlim semaphore = new SemaphoreSlim(6);
//        private OpenDotaApiService service = new OpenDotaApiService();

//        public async Task<List<DotaMatchData>> GetMatchesInRangeAsync(long startMatchId, long endMatchId)
//        {
//            List<DotaMatchData> allMatches = new List<DotaMatchData>();

//            long batchSize = 6;
//            for (long i = startMatchId; i <= endMatchId; i += batchSize)
//            {
//                long endBatchMatchId = Math.Min(i + batchSize - 1, endMatchId);
//                var batchMatches = await GetMatchesInBatchAsync(i, endBatchMatchId);
//                allMatches.AddRange(batchMatches);
//            }

//            return allMatches;
//        }

//        public async Task<List<DotaMatchData>> GetMatchesByMatchIdAsync(List<long> matchesId)
//        {
//            var allMatches = new List<DotaMatchData>();

//            var batchSize = 6;
//            for (var i = 0; i <= matchesId.Count; i += batchSize)
//            {
//                var endBatchMatchId = Math.Min(i + batchSize - 1, matchesId.Count - 1);
//                var batchMatches = await GetMatchesInBatchByMatchIdAsync(matchesId, i, endBatchMatchId);
//                allMatches.AddRange(batchMatches);
//            }

//            return allMatches;
//        }

//        private async Task<List<DotaMatchData>> GetMatchesInBatchAsync(long startMatchId, long endMatchId)
//        {
//            await semaphore.WaitAsync();

//            try
//            {
//                List<Task<DotaMatchData>> matchTasks = new List<Task<DotaMatchData>>();

//                for (long i = startMatchId; i <= endMatchId; i++)
//                {
//                    matchTasks.Add(service.GetMatchAsync(i));
//                }

//                var matches = await Task.WhenAll(matchTasks);
//                return matches.ToList();
//            }
//            finally
//            {
//                semaphore.Release();
//            }
//        }

//        private async Task<List<DotaMatchData>> GetMatchesInBatchByMatchIdAsync(List<long> matchesId,
//            int startMatchId, int endMatchId)
//        {
//            await semaphore.WaitAsync();

//            try
//            {
//                List<Task<DotaMatchData>> matchTasks = new List<Task<DotaMatchData>>();

//                for (var i = startMatchId; i <= endMatchId; i++)
//                {
//                    matchTasks.Add(service.GetMatchAsync(matchesId[i]));
//                }

//                var matches = await Task.WhenAll(matchTasks);
//                return matches.ToList();
//            }
//            finally
//            {
//                semaphore.Release();
//            }
//        }
//    }

//    //AS
//    public class MatchIDsProcessor
//    {
//        private static SemaphoreSlim semaphore = new SemaphoreSlim(6);
//        private OpenDotaApiService service = new OpenDotaApiService();

//        public async Task<List<long>> GetMatchesIdAsync(long matchIdCount)
//        {
//            var allMatchesId = new List<long>();
//            var batchSize = 3;
//            for (long i = 0; i < matchIdCount; i += batchSize)
//            {
//                //Console.WriteLine($"GetMatchesIdAsync {i}");
//                //long endBatchMatchId = Math.Min(i + batchSize - 1, matchIdCount);
//                //var batchMatchesId = await GetMatchesIdInBatchAsync(batchSize);
//                //foreach (var matchId in matchesId)
//                //    yield return matchId;
//                var matchesId = await service.GetMatchIdAsync();
//                if (matchesId is not null)
//                    allMatchesId.AddRange(matchesId);
//            }

//            return allMatchesId;
//        }

//        private async Task<IEnumerable<long>> GetMatchesIdInBatchAsync(long batchSize)
//        {
//            await semaphore.WaitAsync();

//            try
//            {
//                var matchTasks = new List<Task<List<long>>>();

//                for (long i = 0; i <= batchSize; i++)
//                {
//                    matchTasks.Add(service.GetMatchIdAsync());
//                }

//                var matches = await Task.WhenAll(matchTasks);
//                return matches
//                    .Where(matchIds => matchIds != null)
//                    .SelectMany(matchIds => matchIds);
//            }
//            finally
//            {
//                semaphore.Release();
//            }
//        }
//    }

//    public class PickBan
//    {
//        public bool is_pick { get; set; }
//        public int hero_id { get; set; }
//        public int team { get; set; }
//        public int order { get; set; }

//    }

//    public class PlayerPicks
//    {
//        public int hero_id { get; set; }
//        public int team_number { get; set; }

//    }


//    public class Match
//    {
//        public long match_id { get; set; }
//        public bool radiant_win { get; set; }
//        public int duration { get; set; }
//        public int? skill { get; set; }
//        public int region { get; set; }
//        public int game_mode { get; set; }
//        public int lobby_type { get; set; }
//        public object all_word_counts { get; set; }
//        public IList<PickBan> picks_bans { get; set; }
//        public IList<PlayerPicks> players { get; set; }

//        public DotaMatchData ToDotaMatchData()
//        {
//            if (picks_bans == null || game_mode != 22)
//                return null;

//            var radiantTeam = new List<float>();
//            var direTeam = new List<float>();
//            foreach (var pickBan in picks_bans)
//            {
//                if (pickBan.is_pick)
//                {
//                    if (pickBan.team == 0)
//                        radiantTeam.Add(pickBan.hero_id);
//                    else if (pickBan.team == 1)
//                        direTeam.Add(pickBan.hero_id);
//                }
//            }
//            var matchData = new DotaMatchData
//            {
//                RadiantFeatures = radiantTeam.ToArray(),
//                DireFeatures = direTeam.ToArray(),
//                RadiantWin = radiant_win
//            };

//            return matchData;
//        }
//    }
//}
