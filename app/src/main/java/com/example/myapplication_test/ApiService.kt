import com.google.gson.annotations.SerializedName

import android.os.Parcel
import android.os.Parcelable


// gpt api 요청 데이터모델
data class ChatMessage(
    val role: String,
    val content: String
)
data class ChatRequest(
    val messages: List<ChatMessage>,
    val model: String
)


// gpt api 응답 데이터모델
data class ChatCompletionResponse(
    @SerializedName("id") val id: String,
    @SerializedName("object") val objectName: String,
    @SerializedName("created") val created: Long,
    @SerializedName("model") val model: String,
    @SerializedName("system_fingerprint") val systemFingerprint: String,
    @SerializedName("choices") val choices: List<Choice>,
    @SerializedName("usage") val usage: Usage
)
data class Choice(
    @SerializedName("index") val index: Int,
    @SerializedName("message") val message: Message,
    @SerializedName("logprobs") val logprobs: Any?,
    @SerializedName("finish_reason") val finishReason: String
)
data class Message(
    @SerializedName("role") val role: String,
    @SerializedName("content") val content: String
)
data class Usage(
    @SerializedName("prompt_tokens") val promptTokens: Int,
    @SerializedName("completion_tokens") val completionTokens: Int,
    @SerializedName("total_tokens") val totalTokens: Int
)



// tmdb api 응답 영화 데이터모델
data class TmdbSearchResponse(
    @SerializedName("page") val page: Int,
    @SerializedName("results") val results: List<Movie>,
    @SerializedName("total_pages") val total_pages: Int,
    @SerializedName("total_results") val total_results: Int
)
data class Movie(
    @SerializedName("adult") val adult: Boolean,
    @SerializedName("backdrop_path") val backdrop_path: String?,
    @SerializedName("genre_ids") val genre_ids: List<Int>,
    @SerializedName("id") val id: Int,
    @SerializedName("original_language") val original_language: String,
    @SerializedName("original_title") val original_title: String,
    @SerializedName("overview") val overview: String,
    @SerializedName("popularity") val popularity: Double,
    @SerializedName("poster_path") val poster_path: String?,
    @SerializedName("release_date") val release_date: String,
    @SerializedName("title") val title: String,
    @SerializedName("video") val video: Boolean,
    @SerializedName("vote_average") val vote_average: Double,
    @SerializedName("vote_count") val vote_count: Int
)


data class Movielist(
    val title: String,
    val overview: String,
    val poster_path: String,
    val popularitys: String,
    val release_date: String) : Parcelable {

    constructor(parcel: Parcel) : this(
        parcel.readString() ?: "",
        parcel.readString() ?: "",
        parcel.readString() ?: "",
        parcel.readString() ?: "",
        parcel.readString() ?: ""
    )

    override fun writeToParcel(parcel: Parcel, flags: Int) {
        parcel.writeString(title)
        parcel.writeString(overview)
        parcel.writeString(poster_path)
        parcel.writeString(popularitys)
        parcel.writeString(release_date)
    }

    override fun describeContents(): Int {
        return 0
    }

    companion object CREATOR : Parcelable.Creator<Movielist> {
        override fun createFromParcel(parcel: Parcel): Movielist {
            return Movielist(parcel)
        }

        override fun newArray(size: Int): Array<Movielist?> {
            return arrayOfNulls(size)
        }
    }
}


data class Tvlist(
    val title: String,
    val overview: String,
    val poster_path: String,
    val popularitys: String,
    val first_air_date: String) : Parcelable {

    constructor(parcel: Parcel) : this(
        parcel.readString() ?: "",
        parcel.readString() ?: "",
        parcel.readString() ?: "",
        parcel.readString() ?: "",
        parcel.readString() ?: ""
    )

    override fun writeToParcel(parcel: Parcel, flags: Int) {
        parcel.writeString(title)
        parcel.writeString(overview)
        parcel.writeString(poster_path)
        parcel.writeString(popularitys)
        parcel.writeString(first_air_date)
    }

    override fun describeContents(): Int {
        return 0
    }

    companion object CREATOR : Parcelable.Creator<Tvlist> {
        override fun createFromParcel(parcel: Parcel): Tvlist {
            return Tvlist(parcel)
        }

        override fun newArray(size: Int): Array<Tvlist?> {
            return arrayOfNulls(size)
        }
    }
}



// tmdb api 드라마 응답 데이터모델
data class TmdbSearchResponse2(
    @SerializedName("page") val page: Int,
    @SerializedName("results") val results: List<TV>,
    @SerializedName("total_pages") val total_pages: Int,
    @SerializedName("total_results") val total_results: Int
)
data class TV(
    @SerializedName("adult") val adult: Boolean,
    @SerializedName("backdrop_path") val backdrop_path: String?,
    @SerializedName("genre_ids") val genre_ids: List<Int>,
    @SerializedName("id") val id: Int,
    @SerializedName("origin_country") val origin_country: List<String>,
    @SerializedName("original_language") val original_language: String,
    @SerializedName("original_name") val original_name: String,
    @SerializedName("overview") val overview: String,
    @SerializedName("popularity") val popularity: Double,
    @SerializedName("poster_path") val poster_path: String?,
    @SerializedName("first_air_date") val first_air_date: String,
    @SerializedName("name") val name: String,
    @SerializedName("vote_average") val vote_average: Double,
    @SerializedName("vote_count") val vote_count: Int
)


data class PredictionRequest(val text: String)
data class PredictionResponse(val prediction: Int)

