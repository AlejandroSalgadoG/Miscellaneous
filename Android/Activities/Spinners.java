package com.activities;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.graphics.Color;

public class Spinners extends Activity {

	private Spinner spinner;
	private LinearLayout linear;
	private Button button;
	private TextView text;
	private boolean blue;
	private boolean white;
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_spinner);
		
		String[] items = {"White", "Red", "Green", "Blue",
						  "Yellow", "Orrange", "Purple", "Black"};
		
		linear = (LinearLayout) findViewById(R.id.sp_layout);
		button = (Button) findViewById(R.id.sp_button);
		text = (TextView) findViewById(R.id.sp_text);
		 
		spinner = (Spinner) findViewById(R.id.spinner);
		ArrayAdapter<String> adapter = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, items);
		adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
		spinner.setAdapter(adapter);
	}
	
	public void changeBackground(View v){
		String color = (String) spinner.getSelectedItem();
		int pos = spinner.getSelectedItemPosition();
		
		if(blue) {
			button.setBackgroundColor(Color.GREEN);
			blue = false;
		}
		if(white){
			spinner.setBackgroundColor(Color.TRANSPARENT);
			text.setTextColor(Color.BLACK);
			white = false;
		}
		
		text.setText(color);
		
		switch(pos){
			case 0:
				linear.setBackgroundColor(Color.WHITE);
				break;
			case 1:
				linear.setBackgroundColor(Color.RED);
				break;
			case 2:
				linear.setBackgroundColor(Color.GREEN);
				button.setBackgroundColor(Color.BLUE);
				blue = true;
				break;
			case 3:
				linear.setBackgroundColor(Color.BLUE);
				break;
			case 4:
				linear.setBackgroundColor(Color.YELLOW);
				break;
			case 5:
				linear.setBackgroundColor(Color.parseColor("#FFA500"));
				break;
			case 6:
				linear.setBackgroundColor(Color.parseColor("#551A8B"));
				break;
			case 7:
				linear.setBackgroundColor(Color.BLACK);
				spinner.setBackgroundColor(Color.WHITE);
				text.setTextColor(Color.WHITE);
				white = true;
				break;
		}
	}
}
