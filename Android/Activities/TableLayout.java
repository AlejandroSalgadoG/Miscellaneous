package com.activities;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class TableLayout extends Activity {
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_table_layout);
	}
	
	public void setText(View v){
		TextView text = (TextView) findViewById(R.id.numText);
		String num = (String) v.getTag();
		text.setText(num);
	}
}
