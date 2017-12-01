package com.activities;

import com.activities.R.id;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;

public class Gallery extends Activity {
	
	private int[] images = {R.drawable.superman, R.drawable.basketball,
							R.drawable.ironman, R.drawable.cube, 
							R.drawable.linux, R.drawable.mario,
							R.drawable.spiderman, R.drawable.eagle};
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_gallery);
		
		LinearLayout gallery = (LinearLayout) findViewById(id.g_layout);
		
		for(int i=0;i<images.length;i++){
			LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(100,100);
			
			ImageView image = new ImageView(this);
			image.setLayoutParams(params);
			image.setPadding(2, 4, 2, 4);
			image.setImageResource(images[i]);
			image.setScaleType(ImageView.ScaleType.CENTER_INSIDE);
			
			image.setOnClickListener(new View.OnClickListener() {
				@Override
				public void onClick(View v) {
					ImageView main = (ImageView) findViewById(R.id.g_main);
					ImageView change = (ImageView) v;
					main.setImageDrawable(change.getDrawable());
				}
			});
			
			gallery.addView(image);
		}
		
	}
}
