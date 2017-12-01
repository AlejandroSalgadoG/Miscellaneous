package com.activities;

import android.app.ListActivity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.ListView;

public class Main extends ListActivity {
	
	public String[] table = { "LinearLayout", "FrameLayout",
							  "TableLayout", "Buttons",
							  "BooleanControls", "Gallery",
							  "Spinner", "Web"};
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		setListAdapter(new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1 , table));
	}
	
	public void onListItemClick(ListView parent, View v, int pos, long id) {
		Intent intent; 
		
		switch(pos){
			case 0: 
				intent = new Intent(v.getContext(), LinearLayout.class);
				startActivity(intent);
				break;
			case 1:
				intent = new Intent(v.getContext(), FrameLayout.class);
				startActivity(intent);
				break;
			case 2:
				intent = new Intent(v.getContext(), TableLayout.class);
				startActivity(intent);
				break;
			case 3:
				intent = new Intent(v.getContext(), Buttons.class);
				startActivity(intent);
				break;
			case 4:
				intent = new Intent(v.getContext(), BooleanControl.class);
				startActivity(intent);
				break;
			case 5:
				intent = new Intent(v.getContext(), Gallery.class);
				startActivity(intent);
				break;
			case 6:
				intent = new Intent(v.getContext(), Spinners.class);
				startActivity(intent);
				break;
			case 7:
				intent = new Intent(v.getContext(), Web.class);
				startActivity(intent);
				break;
		}
	}
}
