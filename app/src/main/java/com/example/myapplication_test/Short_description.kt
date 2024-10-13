package com.example.myapplication_test

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.short_description.*

class Short_description : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.short_description)

        start_button.setOnClickListener {
            startActivity(Intent(this, Selection::class.java))
            finish()
        }

    }
}