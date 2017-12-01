package com.activities;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;

public class FrameLayout extends Activity {
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_frame_layout);
		
		final LinearLayout lin1 = (LinearLayout) findViewById(R.id.fr_linear1);
		final LinearLayout lin2 = (LinearLayout) findViewById(R.id.fr_linear2);
		
		Button screen1 = (Button) findViewById(R.id.screen1);
		Button screen2 = (Button) findViewById(R.id.screen2);
		
		lin2.setVisibility(View.GONE);
		
		screen1.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				lin1.setVisibility(View.VISIBLE);
				lin2.setVisibility(View.GONE);
			}
		});
		
		screen2.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				lin1.setVisibility(View.GONE);
				lin2.setVisibility(View.VISIBLE);
			}
		});
	}
	
}
