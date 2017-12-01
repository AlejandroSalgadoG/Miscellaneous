package com.activities;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ScrollView;

public class LinearLayout extends Activity {
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_linear_layout);
		
		final ScrollView scroll = (ScrollView) findViewById(R.id.scroll);
		
		Button top = (Button) findViewById(R.id.top);
		Button bottom = (Button) findViewById(R.id.bottom);
		
		top.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				scroll.fullScroll(View.FOCUS_UP);
			}
		});
		
		bottom.setOnClickListener(new View.OnClickListener() {	
			@Override
			public void onClick(View v) {
				scroll.fullScroll(View.FOCUS_DOWN);
			}
		});
	}
	
}
