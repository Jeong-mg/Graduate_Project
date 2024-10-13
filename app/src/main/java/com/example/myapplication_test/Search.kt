package com.example.myapplication_test

import ChatMessage
import ChatRequest
import Movie
import TV
import TmdbSearchResponse
import TmdbSearchResponse2
import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.android.synthetic.main.search.*
import kotlinx.coroutines.*

class Search  : AppCompatActivity() {

    // Results_display로 넘어가는 배열값
    // **리사이클러뷰에 요소 추가시 여기에 추가해야함**
    var titles = mutableListOf<String>()
    var overviews = mutableListOf<String>()
    var posterpaths = mutableListOf<String>()
    var popularity = mutableListOf<String>()
    var release_date = mutableListOf<String>()

    var titles2 = mutableListOf<String>()
    var overviews2 = mutableListOf<String>()
    var posterpaths2 = mutableListOf<String>()
    var popularity2 = mutableListOf<String>()
    var first_air_date = mutableListOf<String>()

    // GPT-3.5 Turbo API 요청에 필요한 데이터 모델
    data class GPTRequest(
        val model: String,
        val prompt: String,
        val maxTokens: Int
    )



    // TMDB API 요청 - 동기화된 코드
    suspend fun communicateWithTM_movie(query: String): Unit? { // 영화
        val includeAdult = false
        val language = "ko-KR" // 한국어
        val page = 1
        val apiKey = "95dbc652cce0ae9d409c0e8feaf97d2b"


        return try {
            val response_movie = RetrofitClient.instance_2.searchMovies(query, includeAdult, language, page, apiKey)
            if (response_movie.isSuccessful) {
                val result = response_movie.body().toString()
                // gpt 응답 값 전체 출력
                Log.d("MyTag", result)


                // 응답문자열 파싱 -> 데이터추출
                fun parseTmdbSearchResponse(responseString: String): TmdbSearchResponse {
                    val regex = Regex("""TmdbSearchResponse\(page=(\d+), results=\[(.*?)\], total_pages=(\d+), total_results=(\d+)\)""")

                    val matchResult = regex.find(responseString)
                        ?: throw IllegalArgumentException("Invalid response string format")

                    val (pageStr, resultsStr, totalPagesStr, totalResultsStr) = matchResult.destructured

                    val page = pageStr.toInt()
                    val totalPages = totalPagesStr.toInt()
                    val totalResults = totalResultsStr.toInt()

                    // tmdb 응답값 예시 )
                    // Movie(adult=false, backdrop_path=/h5pAEVma835u8xoE60kmLVopLct.jpg, genre_ids=[16, 10751, 14], id=16859, original_language=ja, original_title=魔女の宅急便, overview=A young witch, on her mandatory year of independent life, finds fitting into a new community difficult while she supports herself by running an air courier service., popularity=45.235, poster_path=/Aufa4YdZIv4AXpR9rznwVA5SEfd.jpg, release_date=1989-07-29, title=Kiki's Delivery Service, video=false, vote_average=7.818, vote_count=3771)

                    // 결과 목록 파싱
                    val movies = mutableListOf<Movie>()
                    val movieRegex = Regex("""overview=(.*?),.*?popularity=(.*?),.*?poster_path=(.*?),.*?release_date=(.*?),.*?title=(.*?),""")

                    movieRegex.findAll(resultsStr).forEach { movieMatchResult ->
                        val (overview, popularityStr, poster_path, release_date, title) = movieMatchResult.destructured
                        val popularity = popularityStr.toDoubleOrNull() ?: 0.0 // 문자열을 Double로 변환하거나 기본값으로 설정
                        movies.add(Movie(false,
                            "",
                            listOf(14, 16, 10751),
                            8392,
                            "",
                            "",
                            overview,
                            popularity,
                            poster_path,
                            release_date,
                            title,
                            false,
                            8.07,
                            7441))
                    }

                    return TmdbSearchResponse(page, movies, totalPages, totalResults)
                }

                // 응답 문자열을 파싱하여 TmdbSearchResponse 객체 생성
                val tmdbSearchResponse: TmdbSearchResponse = parseTmdbSearchResponse(result)
                tmdbSearchResponse.let { handleTmdbResponse(it) }  // 배열에 값 넣는 함수



            } else {
                null
            }
        } catch (e: Exception) {
            Log.e("MyTag", "Error communicating with TM: ${e.message}", e)
            null
        }


    }

    suspend fun communicateWithTM_tv(query: String): Unit? { // 드라마
        val includeAdult = false
        val language = "ko-KR" // 한국어
        val page = 1
        val apiKey = "95dbc652cce0ae9d409c0e8feaf97d2b"


        return try {
            val response_tv = RetrofitClient.instance_3.searchTvs(query, includeAdult, language, page, apiKey)
            if (response_tv.isSuccessful) {
                val result = response_tv.body().toString()
                // gpt 응답 값 전체 출력
                Log.d("MyTag", result)


                // 응답문자열 파싱 -> 데이터추출
                fun parseTmdbSearchResponse(responseString: String): TmdbSearchResponse2 {
                    val regex = Regex("""TmdbSearchResponse2\(page=(\d+), results=\[(.*?)\], total_pages=(\d+), total_results=(\d+)\)""")

                    val matchResult = regex.find(responseString)
                        ?: throw IllegalArgumentException("Invalid response string format")

                    val (pageStr, resultsStr, totalPagesStr, totalResultsStr) = matchResult.destructured

                    val page = pageStr.toInt()
                    val totalPages = totalPagesStr.toInt()
                    val totalResults = totalResultsStr.toInt()

                    // 결과 목록 파싱
                    val tvs = mutableListOf<TV>()
                    val tvRegex = Regex("""overview=(.*?),.*?popularity=(.*?),.*?poster_path=(.*?),.*?first_air_date=(.*?),.*?name=(.*?),""")

                    tvRegex.findAll(resultsStr).forEach { tvMatchResult ->
                        val (overview, popularityStr, poster_path, first_air_date, name) = tvMatchResult.destructured
                        val popularity = popularityStr.toDoubleOrNull() ?: 0.0 // 문자열을 Double로 변환하거나 기본값으로 설정
                        tvs.add(TV(false,
                            "",
                            listOf(14, 16, 10751),
                            8392,
                            listOf("", "", ""),
                            "",
                            "",
                            overview,
                            popularity,
                            poster_path,
                            first_air_date,
                            name,
                            8.07,
                            7441))
                    }

                    return TmdbSearchResponse2(page, tvs, totalPages, totalResults)
                }

                // 응답 문자열을 파싱하여 TmdbSearchResponse 객체 생성
                val tmdbSearchResponse: TmdbSearchResponse2 = parseTmdbSearchResponse(result)
                tmdbSearchResponse.let { handleTmdbResponse2(it) }  // 배열에 값 넣는 함수



            } else {
                null
            }
        } catch (e: Exception) {
            Log.e("MyTag", "Error communicating with TM: ${e.message}", e)
            null
        }


    }


    // 배열에 값 넣기
    fun handleTmdbResponse(response: TmdbSearchResponse) { // 영화
        response.let {
            for (movie in it.results) {
                titles.add(movie.title)
                overviews.add(movie.overview)
                posterpaths.add(movie.poster_path ?: "")
                popularity.add(movie.popularity.toString())
                release_date.add(movie.release_date)
                // **더 추가 필요**

            }
        }
    }
    fun handleTmdbResponse2(response: TmdbSearchResponse2) { // 드라마
        response.let {
            for (tv in it.results) {
                titles2.add(tv.name)
                overviews2.add(tv.overview)
                posterpaths2.add(tv.poster_path ?: "")
                popularity2.add(tv.popularity.toString())
                first_air_date.add(tv.first_air_date)
                // **더 추가 필요**

            }
        }
    }



    inner class MainLogic(question: String, question2: String) {

        val messages = listOf( // 영화 질문
            ChatMessage(role = "user", content = question)
        )
        val request = ChatRequest(
            messages = messages,
            model = "gpt-3.5-turbo"
        )

        val messages2 = listOf( // 드라마 질문
            ChatMessage(role = "user", content = question2)
        )
        val request2 = ChatRequest(
            messages = messages2,
            model = "gpt-3.5-turbo"
        )

        val intent = Intent(this@Search, Results_display::class.java)

        // API와 통신하는 함수 - 동기화된 코드
        suspend fun communicateWithAPI(requestData: GPTRequest) {
            // 영화
            try {
                val response = RetrofitClient.instance.getCompletion(request)
                if (response.isSuccessful) {
                    val result = response.body()
                    // gpt 응답 값 전체 출력
                    Log.d("MyTag", result.toString())

                    val movieTitles = result?.choices!![0].message.content.split("\n")
                        .map { it.substringAfter(". ").trim() } // 각 줄에서 번호와 영화 제목을 분리하고 영화 제목만 추출
                        .toTypedArray()

                    // 코루틴 이용해서 동기화
                    val deferredList = movieTitles.map { query ->
                        lifecycleScope.async {
                            communicateWithTM_movie(query)
                        }
                    }

                    // 모든 communicateWithTM 호출이 완료될 때까지 대기
                    deferredList.forEach { deferred ->
                        try {
                            deferred.await()
                        } catch (e: CancellationException) {
                            Log.e("MyTag", "Coroutine was cancelled", e)
                        }
                    }


                    // 모든 호출이 완료된 후에 로그를 출력
                    Log.d("MyTag", "타이틀 사이즈: "+titles.size) // 결과 정상
                    Log.d("MyTag", "타이틀 내용: $titles") // 결과 정상


                    if (titles.size >= 0) {
                        intent.putStringArrayListExtra("titles", ArrayList(titles))
                        intent.putStringArrayListExtra("overviews", ArrayList(overviews))
                        intent.putStringArrayListExtra("posterpaths", ArrayList(posterpaths))
                        intent.putStringArrayListExtra("popularitys", ArrayList(popularity))
                        intent.putStringArrayListExtra("release_dates", ArrayList(release_date))

                    }
                } else {
                    val errorMessage = response.errorBody()?.string()
                    Log.d("MyTag", "Failed: $errorMessage")
                }
            } catch (e: Exception) {
                Log.e("MyTag", "Error communicating with API: ${e.message}", e)
            }

            // 드라마
            try {
                val response = RetrofitClient.instance.getCompletion(request2)
                if (response.isSuccessful) {
                    val result = response.body()
                    // gpt 응답 값 전체 출력
                    Log.d("MyTag", result.toString())

                    val tvTitles = result?.choices!![0].message.content.split("\n")
                        .map { it.substringAfter(". ").trim() } // 각 줄에서 번호와 드라마 제목을 분리하고 드라마 제목만 추출
                        .toTypedArray()

                    // 코루틴 이용해서 동기화
                    val deferredList = tvTitles.map { query ->
                        lifecycleScope.async {
                            communicateWithTM_tv(query)
                        }
                    }

                    // 모든 communicateWithTM 호출이 완료될 때까지 대기
                    deferredList.forEach { deferred ->
                        try {
                            deferred.await()
                        } catch (e: CancellationException) {
                            Log.e("MyTag", "Coroutine was cancelled", e)
                        }
                    }


                    // 모든 호출이 완료된 후에 로그를 출력
                    Log.d("MyTag", "타이틀 사이즈: "+titles2.size) // 결과 정상
                    Log.d("MyTag", "타이틀 내용: $titles2") // 결과 정상


                    if (titles2.size >= 0) {
                        intent.putStringArrayListExtra("titles2", ArrayList(titles2))
                        intent.putStringArrayListExtra("overviews2", ArrayList(overviews2))
                        intent.putStringArrayListExtra("posterpaths2", ArrayList(posterpaths2))
                        intent.putStringArrayListExtra("popularitys2", ArrayList(popularity2))
                        intent.putStringArrayListExtra("first_air_dates", ArrayList(first_air_date))

                        startActivity(intent) // Results_display 화면으로 넘어감
                        finish()
                    }
                } else {
                    val errorMessage = response.errorBody()?.string()
                    Log.d("MyTag", "Failed: $errorMessage")
                }
            } catch (e: Exception) {
                Log.e("MyTag", "Error communicating with API: ${e.message}", e)
            }
        }
    }



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.search)

        back_button.setOnClickListener {
            startActivity(Intent(applicationContext, Selection::class.java))
            finish()
        }


        // GPT-3.5 Turbo API 요청 데이터 설정
        val requestData = GPTRequest(
            model = "gpt-3.5-turbo-0613",
            prompt = "Once upon a time",
            maxTokens = 50
        )


        //메인 기능 실행
        go_button.setOnClickListener {

            // question: 사용자가 직접 입력한 '검색'관련 문장
            val question = et_msg?.text.toString()
            et_msg.setText("")
            val appendedText_movie = "아무런 코멘트 없이 오로지 영화제목만 10개 말해줘."
            val questionText_movie = "$question $appendedText_movie"

            val appendedText_tv = "아무런 코멘트 없이 오로지 드라마제목만 10개 말해줘."
            val questionText_tv = "$question $appendedText_tv"

            Log.d("MyTag", questionText_movie)
            Log.d("MyTag", questionText_tv)



            // API와 통신
            val mainLogic = MainLogic(questionText_movie, questionText_tv)

            GlobalScope.launch {
                mainLogic.communicateWithAPI(requestData)
            }

        }

    }

}