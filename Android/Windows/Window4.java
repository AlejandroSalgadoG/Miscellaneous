package com.windows;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class Window4 extends Activity{
	
	private TextView text;
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_window4);
		
		text = (TextView) findViewById(R.id.path);
		text.setText(Path.getPath());
	}

	public void back(View v){
		Path.replacePath("Window4 \n");
		finish();
	}

	public void changeWindow2(View v){
		Intent intent = new Intent(Window4.this, Window2.class);
		Path.setPath("Window2 \n");
	    startActivity(intent);
	}
	
	public void changeWindow3(View v){
		Intent intent = new Intent(Window4.this, Window3.class);
		Path.setPath("Window3 \n");
	    startActivity(intent);
	}
	
	public void changeWindow5(View v){
		Intent intent = new Intent(Window4.this, Window5.class);
		Path.setPath("Window5 \n");
	    startActivity(intent);
	}
}
