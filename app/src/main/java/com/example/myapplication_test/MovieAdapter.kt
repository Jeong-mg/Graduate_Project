package com.example.myapplication_test

import Movielist
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.squareup.picasso.Picasso
import kotlinx.android.synthetic.main.results_display_list.view.overviewTextView
import kotlinx.android.synthetic.main.results_display_list.view.posterImageView
import kotlinx.android.synthetic.main.results_display_list.view.titleTextView

class MovieAdapter(private val movies: List<Movielist>, private val listener: OnItemClickListener) : RecyclerView.Adapter<MovieAdapter.ViewHolder>() {

    private var itemClickListner: OnItemClickListener = listener // listener로 초기화


    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView), View.OnClickListener {
        // tmdb 응답값 설명
        /*
    adult: 해당 영화가 성인용으로 지정되었는지를 나타내는 부울 값입니다.
    backdrop_path: 영화 배경 화면의 이미지 경로를 나타냅니다.
    genre_ids: 해당 영화의 장르 식별자(ID) 목록을 나타냅니다.
    id: 해당 영화의 고유 식별자(ID)를 나타냅니다.
    original_language: 해당 영화의 원본 언어를 나타냅니다.
    original_title: 해당 영화의 원본 제목을 나타냅니다.
    overview: 해당 영화의 개요(줄거리)를 나타냅니다.
    popularity: 해당 영화의 인기도를 나타내는 수치입니다.
    poster_path: 영화 포스터의 이미지 경로를 나타냅니다.
    release_date: 해당 영화의 개봉일을 나타냅니다.
    title: 해당 영화의 제목을 나타냅니다.
    video: 해당 영화에 비디오가 있는지 여부를 나타내는 부울 값입니다.
    vote_average: 해당 영화의 평균 평점을 나타내는 수치입니다.
    vote_count: 해당 영화에 대한 평가(투표) 수를 나타내는 정수 값입니다.

    */


        // results_display_list 요소와 연결
        val textTitle_movie: TextView = itemView.titleTextView
        val textOverview_movie: TextView = itemView.overviewTextView
        val imagePoster_movie: ImageView = itemView.posterImageView
        // **리스트 요소 추가 필요 **

        // 리스트 클릭
        var list_page: LinearLayout = itemView.findViewById(R.id.list_page)


        init {
            itemView.setOnClickListener(this)
        }
        override fun onClick(view: View) {
            val position = adapterPosition
            if (position != RecyclerView.NO_POSITION) {
                val movie = movies[position]
                listener.onItemClick(movie) // 아이템 클릭 리스너 호출 -> 클릭된 아이템의 정보를 MovieDetail에 전달
            }
        }

    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.results_display_list, parent, false)
        return ViewHolder(view)
    }


    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val movie = movies[position]

        holder.textTitle_movie.text = movie.title
        holder.textOverview_movie.text = movie.overview

        // 포스터이미지 로드
        Log.d("MyTag", movie.poster_path)

        // **근데 이미지 한개만 일괄적으로 뜨는 문제발생함 수정필요** 초기화문제???
        if (!movie.poster_path.isNullOrBlank()) {
            val posterUrl = "https://image.tmdb.org/t/p/w500${movie.poster_path}"
            Picasso.get().load(posterUrl).into(holder.imagePoster_movie)
        }
        //else{ // **예외처리해야함**
        //    holder.imagePoster.setImageResource(R.drawable.placeholder_poster)
        //}



        // 리사이클러뷰 아이템클릭 리스너 (이웃 리스트 세부정보 확인)
        holder.list_page.setOnClickListener {
            itemClickListner.onItemClick(movie)
        }

    }


    // 리스트 클릭
    interface OnItemClickListener {
        fun onItemClick(movie: Movielist) // 영화
    }


    override fun getItemCount(): Int {
        return movies.size
    }


}
