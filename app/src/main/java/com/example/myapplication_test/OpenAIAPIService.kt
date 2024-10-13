import retrofit2.Call
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Headers
import retrofit2.http.POST
import retrofit2.http.Query

interface OpenAIAPIService {
    @Headers(
        "Content-Type: application/json",
        "Authorization: Bearer "
    )
    @POST("chat/completions")
    suspend fun getCompletion(@Body requestBody: ChatRequest): Response<ChatCompletionResponse>
}

interface TMDBService {
    @GET("search/movie")
    suspend fun searchMovies(
        @Query("query") query: String,
        @Query("include_adult") includeAdult: Boolean,
        @Query("language") language: String,
        @Query("page") page: Int,
        @Query("api_key") apiKey: String
    ): Response<TmdbSearchResponse>
}

interface TMDBTVService {
    @GET("search/tv")
    suspend fun searchTvs(
        @Query("query") query: String,
        @Query("include_adult") includeAdult: Boolean,
        @Query("language") language: String,
        @Query("page") page: Int,
        @Query("api_key") apiKey: String
    ): Response<TmdbSearchResponse2>
}

interface KorBERTServer {
    @Headers("Content-Type: application/json")
    @POST("predict")
    fun getPrediction(@Body request: PredictionRequest): Call<PredictionResponse>
}