package com.example.myapplication_test

import Tvlist
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.drawable.Drawable
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.squareup.picasso.Picasso
import kotlinx.android.synthetic.main.movie_detail.detail_posterImageView
import kotlinx.android.synthetic.main.search.back_button

class TvDetail : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.tv_detail)

        back_button.setOnClickListener {
            startActivity(Intent(applicationContext, Selection::class.java))
            finish()
        }


        val tv = intent.getParcelableExtra<Tvlist>("tv")
        tv?.let {
            findViewById<TextView>(R.id.detail_titleTextView).text = it.title // 타이틀 설정
            findViewById<TextView>(R.id.detail_overviewTextView).text = it.overview // 미리뷰 설정
            findViewById<TextView>(R.id.detail_popularityTextView).text = it.popularitys // 인기도 설정
            findViewById<TextView>(R.id.detail_firstairdateTextView).text = it.first_air_date // 방영일 설정

            // 포스터이미지 설정
            val topColor = Color.parseColor("#FF000D6B") // 투명한 색상
            val bottomColor = Color.parseColor("#00000000") // 불투명한 색상

            val frameLayout = FrameLayout(this@TvDetail)
            val layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
            frameLayout.layoutParams = layoutParams

            // 포스터이미지 설정
            if (!it.poster_path.isNullOrBlank()) {
                val posterUrl = "https://image.tmdb.org/t/p/w500${it.poster_path}"
                Picasso.get().load(posterUrl).into(object : com.squareup.picasso.Target {
                    override fun onBitmapLoaded(bitmap: Bitmap?, from: Picasso.LoadedFrom?) {
                        // 이미지 로드 성공 시, 이미지 뷰에 이미지 설정
                        val imageView = ImageView(this@TvDetail)
                        imageView.layoutParams = ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT, // 가로 길이를 부모에 맞게 설정
                            ViewGroup.LayoutParams.MATCH_PARENT  // 세로 길이를 부모에 맞게 설정
                        )

                        imageView.setImageBitmap(bitmap)
                        imageView.scaleType = ImageView.ScaleType.CENTER_CROP

                        frameLayout.addView(imageView)

                        // 포스터 이미지 위에 그라데이션 효과를 적용
                        val gradientView = View(this@TvDetail)
                        gradientView.layoutParams = ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT
                        )
                        val gradientDrawable = GradientDrawable(
                            GradientDrawable.Orientation.TOP_BOTTOM,
                            intArrayOf(topColor, bottomColor)
                        )
                        gradientView.background = gradientDrawable

                        frameLayout.addView(gradientView)

                    }



                    override fun onBitmapFailed(e: Exception?, errorDrawable: Drawable?) {
                        // 이미지 로드 실패 시 처리할 내용
                    }
                    override fun onPrepareLoad(placeHolderDrawable: Drawable?) {
                        // 이미지 로딩 전 처리할 내용
                    }
                })
            } else {
                // 포스터 이미지가 없을 경우 기본 배경 이미지를 설정
                // linearLayout.setBackgroundResource(R.drawable.default_background)
            }

            detail_posterImageView.addView(frameLayout)

        }


    }
}
