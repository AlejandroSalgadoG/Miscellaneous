package com.windows;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class Window2 extends Activity {
	
	private TextView text;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_window2);
		
		text = (TextView) findViewById(R.id.path);
		text.setText(Path.getPath());
	}
	
	public void back(View v){
		Path.replacePath("Window2 \n");
		finish();
	}

	public void changeWindow3(View v){
		Intent intent = new Intent(Window2.this, Window3.class);
		Path.setPath("Window3 \n");
	    startActivity(intent);
	}
	
	public void changeWindow4(View v){
		Intent intent = new Intent(Window2.this, Window4.class);
		Path.setPath("Window4 \n");
	    startActivity(intent);
	}
	
	public void changeWindow5(View v){
		Intent intent = new Intent(Window2.this, Window5.class);
		Path.setPath("Window5 \n");
	    startActivity(intent);
	}
}
