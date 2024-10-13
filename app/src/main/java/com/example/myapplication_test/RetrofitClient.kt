
import com.jakewharton.retrofit2.adapter.kotlin.coroutines.CoroutineCallAdapterFactory
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object RetrofitClient {
    // gpt api
    private const val BASE_URL = "https://api.openai.com/v1/"

    val instance: OpenAIAPIService by lazy {
        val retrofit = Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        retrofit.create(OpenAIAPIService::class.java)
    }


    // tmdb api
    private const val BASE_URL_2 = "https://api.themoviedb.org/3/"

    // 영화
    val instance_2: TMDBService by lazy {
        val retrofit = Retrofit.Builder()
            .baseUrl(BASE_URL_2)
            .addConverterFactory(GsonConverterFactory.create()) // Gson 컨버터 팩토리 추가
            .addCallAdapterFactory(CoroutineCallAdapterFactory()) // 코루틴 추가
            .build()

        retrofit.create(TMDBService::class.java)
    }

    // 드라마
    val instance_3: TMDBTVService by lazy {
        val retrofit = Retrofit.Builder()
            .baseUrl(BASE_URL_2)
            .addConverterFactory(GsonConverterFactory.create()) // Gson 컨버터 팩토리 추가
            .addCallAdapterFactory(CoroutineCallAdapterFactory()) // 코루틴 추가
            .build()

        retrofit.create(TMDBTVService::class.java)
    }



    // ** KorBERT - flask server **
    private const val BASE_URL_3 = "http://127.0.0.1:5000/todos"

    val instance_4: KorBERTServer by lazy {
        val retrofit = Retrofit.Builder()
            .baseUrl(BASE_URL_3)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        retrofit.create(KorBERTServer::class.java)
    }


}