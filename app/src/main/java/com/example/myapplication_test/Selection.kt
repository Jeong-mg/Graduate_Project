package com.example.myapplication_test

import android.animation.Animator
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.selection.*
import kotlinx.android.synthetic.main.short_description.*

class Selection  : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.selection)

        //초기 위치 설정
        ballon1.translationY = 1000f
        //처음에 투명하게 설정
        ballon1.alpha = 0f
        // ObjectAnimator를 사용하여 alpha 속성을 변경하여 나타나는 애니메이션 생성
        val translateYAnimator1 = ObjectAnimator.ofFloat(ballon1, "translationY", 1000f, 0f)
        translateYAnimator1.duration = 2000 // 애니메이션 지속 시간 (밀리초)
        val alphaAnimator1 = ObjectAnimator.ofFloat(ballon1, "alpha", 0f, 1f)
        alphaAnimator1.duration = 2000 // 애니메이션 지속 시간 (밀리초)

        val animatorSet1 = AnimatorSet()
        animatorSet1.playTogether(translateYAnimator1, alphaAnimator1)

        // 애니메이션 시작
        animatorSet1.start()



        //초기 위치 설정
        ballon2.translationY = 1000f
        //처음에 투명하게 설정
        ballon2.alpha = 0f
        // ObjectAnimator를 사용하여 alpha 속성을 변경하여 나타나는 애니메이션 생성
        val translateYAnimator2 = ObjectAnimator.ofFloat(ballon2, "translationY", 1000f, 0f)
        translateYAnimator2.duration = 2000 // 애니메이션 지속 시간 (밀리초)
        translateYAnimator2.startDelay = 2000 // 애니메이션 시작 전 대기 시간 (밀리초)
        val alphaAnimator2 = ObjectAnimator.ofFloat(ballon2, "alpha", 0f, 1f)
        alphaAnimator2.duration = 2000 // 애니메이션 지속 시간 (밀리초)
        alphaAnimator2.startDelay = 2000 // 애니메이션 시작 전 대기 시간 (밀리초)

        val animatorSet2 = AnimatorSet()
        animatorSet2.playTogether(translateYAnimator2, alphaAnimator2)

        // 애니메이션 시작
        animatorSet2.start()



        //기분 or 검색 택1 버튼
        mood_button.setOnClickListener {
            startActivity(Intent(applicationContext, Mood::class.java))
            finish()
        }
        search_button.setOnClickListener {
            startActivity(Intent(applicationContext, Search::class.java))
            finish()
        }
    }
}