package com.aplication1;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

public class MainActivity extends Activity {

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		final LinearLayout linearLayout = (LinearLayout) findViewById(R.id.linearLayout);
		
		Button button1 = (Button) findViewById(R.id.button1);
		Button button2 = (Button) findViewById(R.id.button2);
		Button button3 = (Button) findViewById(R.id.button3);
		
		button1.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				ImageView image = new ImageView(getBaseContext());
				image.setImageResource(R.drawable.ic_launcher);
				image.setId(1);
				linearLayout.addView(image);
			}
		});
		
		button2.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				ImageView image = (ImageView) findViewById(1);	
				if(image != null) linearLayout.removeView(image);
			}
		});
		
		button3.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				TextView text = (TextView) findViewById(R.id.text);
				EditText edit = (EditText) findViewById(R.id.edit);
				text.setText(edit.getText().toString());
			}
		});
	}
}
