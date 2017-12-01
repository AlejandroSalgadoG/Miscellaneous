package com.inmortal;

import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

public class Inmortal extends Activity {
	
	public String comment = "";

	@SuppressWarnings("static-access")
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_inmortal);
		
		final InputMethodManager key = (InputMethodManager) getSystemService(getBaseContext().INPUT_METHOD_SERVICE);
		
		SharedPreferences oSettings = getSharedPreferences("app_settings", 0);
		comment = oSettings.getString("screen_msg", ""); 
		
		final LinearLayout linearLayout = (LinearLayout) findViewById(R.id.layout);
		final EditText edit = (EditText) findViewById(R.id.edit);
		final TextView text = new TextView(getBaseContext());
		
		text.setTextAppearance(this, R.style.font);
		
		text.setText(comment);
		linearLayout.addView(text);
		
		Button button = (Button) findViewById(R.id.button);
		
		button.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				key.hideSoftInputFromWindow(text.getWindowToken(),0);
				if(!edit.getText().toString().equals("")){
					comment += edit.getText().toString()+"\n";
					edit.setText("");
					text.setText(comment);
				}
			}
		});
	}
	
	protected void onStop(){
		super.onStop();
		
		SharedPreferences preferences = getSharedPreferences("app_settings", 0);
		SharedPreferences.Editor editor = preferences.edit();
		editor.putString("screen_msg", comment);
		editor.commit();
	}
}
