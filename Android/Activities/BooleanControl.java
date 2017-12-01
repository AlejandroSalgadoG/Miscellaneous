package com.activities;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.CheckBox;
import android.widget.RadioButton;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

public class BooleanControl extends Activity {

	private RadioButton rb1; 
	private RadioButton rb3;
	private CheckBox cb1;
	private CheckBox cb2;
	private ToggleButton tb;
	private Switch sw;
	private TextView text;
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_boolean_control);
		
		rb1 = (RadioButton) findViewById(R.id.bool_radio11);
		rb3 = (RadioButton) findViewById(R.id.bool_radio21);
		
		cb1 = (CheckBox) findViewById(R.id.bool_checkBox1);
		cb2 = (CheckBox) findViewById(R.id.bool_checkBox2);
		
		tb = (ToggleButton) findViewById(R.id.bool_toggleButton);
		
		sw = (Switch) findViewById(R.id.bool_switch);
		
		text = (TextView) findViewById(R.id.bool_text);
		
		cb2.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				Toast.makeText(getBaseContext(), "check button 2 used", Toast.LENGTH_SHORT).show();
			}
		});
	}
	
	public void onClic(View v){
		String data = "";
		
		if(rb1.isChecked()) data += "Option 1 \n";
		else data += "Option 2 \n";
		
		if(rb3.isChecked()) data += "Verdadero \n";
		else data += "Falso \n";
		
		if(cb1.isChecked()) data += "Option A \n";
		
		if(cb2.isChecked()) data += "Option B \n";
		
		if(tb.isChecked()) data += "Activated \n";
		else data += "Deactivated \n";
		
		if(sw.isChecked()) data += "Si \n";
		else data += "No \n";
		
		text.setText(data);
	}
	
}
