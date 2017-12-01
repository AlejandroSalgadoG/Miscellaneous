package com.activities;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

public class Buttons extends Activity {
	
	private boolean pressed;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_buttons);
		
		Button button1 = (Button) findViewById(R.id.btn1);
		Button button2 = (Button) findViewById(R.id.btn2);
		Button button3 = (Button) findViewById(R.id.btn3);
		Button button4 = (Button) findViewById(R.id.btn4);
		
		button1.setOnLongClickListener(new View.OnLongClickListener() {
			@Override
			public boolean onLongClick(View v) {
				Toast.makeText(getBaseContext(), "Long clicked on button # 1", Toast.LENGTH_SHORT).show();
				pressed = true;
				return false;
			}
		});
		
		button2.setOnLongClickListener(new View.OnLongClickListener() {
			@Override
			public boolean onLongClick(View v) {
				Toast.makeText(getBaseContext(), "Long clicked on button # 2", Toast.LENGTH_SHORT).show();
				pressed = true;
				return false;
			}
		});
		
		button3.setOnLongClickListener(new View.OnLongClickListener() {
			@Override
			public boolean onLongClick(View v) {
				Toast.makeText(getBaseContext(), "Long clicked on button # 3", Toast.LENGTH_SHORT).show();
				pressed = true;
				return false;
			}
		});
		
		button4.setOnLongClickListener(new View.OnLongClickListener() {
			@Override
			public boolean onLongClick(View v) {
				Toast.makeText(getBaseContext(), "Long clicked on button # 4", Toast.LENGTH_SHORT).show();
				pressed = true;
				return false;
			}
		});
	}
	
	public void sendMessage(View v){
		if(!pressed){
			String number = (String) v.getTag();
			Toast.makeText(getBaseContext(), "Clicked on button # "+number, Toast.LENGTH_SHORT).show();
		}
		pressed = false; 
	}
}
